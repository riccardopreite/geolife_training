from datetime import datetime
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import sklearn as sk
import pytz
from math import sqrt, cos, radians
from typing import Dict, List, Tuple
from geopy.distance import great_circle
from geopy.point import Point

MyPointTuple = Tuple[float, float, float, datetime] # latitude, longitude, altitude, collection date
StayPointTuple = Tuple[float, float, datetime, datetime] # latitude, longitude, time of arrival, time of leave
RegionTuple = Tuple[float, float] # latitude, longitude


DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
TZ = pytz.timezone('GMT')


class MyPoint(object):
    def __init__(self, latitude: float, longitude: float, altitude: float, collection_date: datetime) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.collection_date = collection_date
    
    def to_tuple(self) -> MyPointTuple:
        return self.latitude, self.longitude, self.altitude, self.collection_date


class StayPoint(object):
    def __init__(self, latitude: float, longitude: float, time_of_arrival: datetime, time_of_leave: datetime) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.time_of_arrival = time_of_arrival
        self.time_of_leave = time_of_leave
        self.region_id = -2

    def to_tuple(self) -> StayPointTuple:
        return self.latitude, self.longitude, self.time_of_arrival, self.time_of_leave, self.region_id


class Region(object):
    def __init__(self, latitude: float, longitude: float) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude

    def to_tuple(self) -> RegionTuple:
        return self.latitude, self.longitude


def convert_row_to_point(row) -> MyPoint:
    converted_date = datetime.strptime(row['time'], DATE_FORMAT)
    gmt_date = TZ.localize(converted_date)
    return MyPoint(row['lat'], row['lon'], row['alt'], gmt_date)

def distance(p1: MyPoint, p2: MyPoint) -> float:
    # Cannot set altitude, geopy does not support it.
    gp_p1 = Point(latitude=p1.latitude, longitude=p1.longitude, altitude=0)
    gp_p2 = Point(latitude=p2.latitude, longitude=p2.longitude, altitude=0)

    plain_distance = great_circle(gp_p1, gp_p2).meters
    altitude_delta = (gp_p1.altitude - gp_p2.altitude)

    return sqrt(plain_distance**2 + altitude_delta**2)

def timedelta(p1: MyPoint, p2: MyPoint) -> float:
    return float((p2.collection_date - p1.collection_date).seconds)

def get_centroid_coordinates(SP, r):
    pass

def get_neighbor_grids(g, G):
    pass

def grid_division(d: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    d is the side length of a part of the grid.

    """
    # globe_max_coordinates = ((90.0, 180.0), (90.0, -180.0), (-90.0, -180.0), (-90.0, 180.0)) # NE, NW, SW, SE

    delta = (d * 0.001) / 111

    """
    The following would be correct for creating a grid of the whole globe but it is impossible from a RAM point of view.
    We must get the ranges in which latitude and longitude values exist in the dataset: 02_find_lat_lon_value_ranges.py

    Results obtained:
    - Max latitude in dataset:  64.751993
    - Min latitude in dataset:  1.044024
    - Max longitude in dataset:  179.9969416
    - Min longitude in dataset:  121.739333333333
    - Latitude ranges: [1,65]
    - Longitude ranges:
        [2,3],
        [4,6],
        [7,13],
        [24,25],
        [37,38],
        [75,131],
        [133,134],
        [135,136],
        [142,143],
        [149,150],
        [174,180],
        [-180,-121],
        [-119,-117],
        [-116,-115],
        [-113,-112],
        [-94,-93],
        [-88,-87],
        [-78,-77],
        [-72,-70],
        [-4,-3]
    """
    # latitude_range_unique = [1,65]
    longitude_range_unique = [
        [2,3],
        [4,6],
        [7,13],
        [24,25],
        [37,38],
        [75,131],
        [133,134],
        [135,136],
        [142,143],
        [149,150],
        [174,180],
        [-180,-121],
        [-119,-117],
        [-116,-115],
        [-113,-112],
        [-94,-93],
        [-88,-87],
        [-78,-77],
        [-72,-70],
        [-4,-3]
    ]
    latitude_range: np.ndarray = np.arange(+65, 1, -delta)


    longitude_range: np.ndarray = np.arange(
        longitude_range_unique[0][0],
        longitude_range_unique[0][1],
        +delta) #+delta
    for lon_range in longitude_range_unique[1:]:
        new_lon_grid = np.arange(
            lon_range[0],
            lon_range[1],
            +delta)
        longitude_range = np.concatenate((longitude_range, new_lon_grid))

    print("Latitude Range x Longitude Range: ", (len(latitude_range) * len(longitude_range)))
    return latitude_range, longitude_range

def assign_stay_points_to_grid_cell(G_lat: np.ndarray, G_lon: np.ndarray) -> Dict[str, List[int]]:
    G_lat_len = len(G_lat)
    G_lon_len = len(G_lon)
    G: Dict[str, List[int]] = {}
    G_lat_i = 0

    while G_lat_i < (G_lat_len - 1):
        G_lon_i = 0
        while G_lon_i < (G_lon_len - 1):
            # These 4 variables represent the grid square in position (G_lat_i, G_lon_i)
            top_left = G_lat[G_lat_i], G_lon[G_lon_i]
            top_right = G_lat[G_lat_i], G_lon[G_lon_i + 1]
            bottom_left = G_lat[G_lat_i + 1], G_lon[G_lon_i]
            bottom_right = G_lat[G_lat_i + 1], G_lon[G_lon_i + 1]

            G_key = str(top_left) + ";" + str(top_right) + ";" + str(bottom_left) + ";" + str(bottom_right)

            SP_len = len(SP)
            sp_index = 0
            while sp_index < SP_len:
                sp = SP[sp_index] # sp[0], sp[1] are respectively latitude, longitude
                if sp.region_id == -2:
                    """

                        TL ---- TR
                        |        |
                        |   sp   |
                        |        | 
                        BL ---- BR
                    
                    """
                    
                    """
                    Equivalent checks:
                    - (sp[0] >= bottom_left[0] and sp[0] <= top_left[0]) is the same as (sp[0] >= bottom_right[0] and sp[0] <= top_right[0])
                    - (sp[1] >= bottom_left[1] and sp[1] <= bottom_right[1]) is the same as (sp[1] >= top_left[1] and sp[1] <= top_right[1])
                    Other combinations exist from those written above.
                    """ 
                    if (sp[0] >= bottom_left[0] and sp[0] <= top_left[0]) and (sp[1] >= bottom_left[1] and sp[1] <= bottom_right[1]):
                        """
                        sp is in the grid element of G with the corner coordinates (top_left, top_right, bottom_left, bottom_right).
                        Therefore:
                        - we generate a key for G (G_key)
                        - we generate a new empty list if the key did not exist previously in G
                        - we append the sp_index to G[G_key]
                        # - we remove sp from SP for freeing the memory (hence we decrement sp_index)
                        """
                        
                        if G_key not in G.keys():
                            G[G_key] = list()
                        G[G_key].append(sp_index)
                        # SP.pop(sp_index)
                        # sp_index -= 1
                        sp.region_id = -1
                sp_index += 1
                # END while sp_index < SP_len
            G_lon_i += 1
            # END while G_lon_i < (G_lon_len - 1)
        G_lat_i += 1
        # END while G_lat_i < (G_lat_len - 1)
    
    return G


def stay_point_detection(traj_k: pd.DataFrame, d_thresh: float, t_thresh: float) -> List[StayPoint]:
    """
    Conditions are that point must be far at most ( <= ) d_thresh and long at least ( >= ) t_thresh
    """
    stay_points: List[StayPoint] = list()

    num_rows = len(traj_k)
    i = 0
    while i < num_rows:
        # latitude = y, longitude = x
        row_i = traj_k.loc[i]
        p_i: MyPoint = convert_row_to_point(row_i)
        j = i + 1
        current_points: List[MyPoint] = list()
        current_points.append(p_i)
        keep = True
        while j < num_rows and keep:
            row_j = traj_k.loc[j]
            p_j: MyPoint = convert_row_to_point(row_j)
            d_ij = distance(p1=p_i, p2=p_j)
            if d_ij <= d_thresh:
                # Points p_i and p_j were collected nearby, so we keep going.
                current_points.append(p_j)
                # if dt_ij >= t_thresh:
                j += 1
            else: # d_ij > d_thresh
                # Points are not nearby, let's check if points collected before p_j
                # were collected in a long time frame i.e >= t_thresh
                last_dt_ij = timedelta(p1=p_i, p2=current_points[-1])
                if last_dt_ij >= t_thresh:
                    # Okay, current_points is a list of points collected near a stay point.
                    # Start again from i = j

                    stay_point: StayPoint = StayPoint(
                        np.average([cp[0] for cp in current_points]),
                        np.average([cp[1] for cp in current_points]),
                        current_points[0][3],
                        current_points[-1][3]
                    )
                    stay_points.append(stay_point)
                    i = j - 1
                keep = False
        i += 1
    return stay_points


def assign_region(G: Dict[str, List[int]]):
    unassigned_stay_points_count: List[Tuple[str, int]] = list()
    for G_key, G_val in G.items():
        unassigned_stay_points_count.append(tuple(G_key, len(G_val)))
    unassigned_stay_points_count.sort(key=lambda t: t[1], reverse=True)

    


def extract_stay_region(trajectories: pd.DataFrame, users, d_thresh: float, t_thresh: float, d: float) -> list:
    """
    trajectories --> phi
    """
    print("start creating sp")
    for u_k in users[:1]:
        traj_k = trajectories[trajectories['user'] == u_k]
        S_k: List[StayPoint] = stay_point_detection(traj_k, d_thresh, t_thresh)
        SP.extend(S_k)
        print("len(traj_k) = ", len(traj_k), ", len(S_k) = ", len(S_k), ", len(SP) = ", len(SP))
    print("created sp")
    #del trajectories
    G_lat, G_lon = grid_division(d)
    print("created grid lat lon")
    # latitude = y, longitude = x
    G = assign_stay_points_to_grid_cell(G_lat, G_lon)
    print("number of grid cells: ", len(G))
    print("number of sp in 1st grid: ", len(G[0]))
    print("sp of first grid:")
    print(G[0])
    
    

"""
List of generated stay points.
"""
SP: List[StayPoint] = list()

"""
List of regions
"""
R = list()

def main():
    input_file: str = 'geolife_trajectories_complete.csv'
    d_thresh: float = 100 # Meters
    t_thresh: float = 300 # Seconds
    d: float = 600
    df_trajectories = pd.read_csv(input_file)
    users = np.unique(df_trajectories['user']) # users --> U
    # print("lat_max = ", df_trajectories['lat'].max(), "lat_min = ", df_trajectories['lat'].min())
    # print("lon_max = ", df_trajectories['lon'].max(), "lon_min = ", df_trajectories['lon'].min())
    # grid_division(d)
    extract_stay_region(df_trajectories, users, d_thresh, t_thresh, d)

if __name__ == '__main__':
    main()
