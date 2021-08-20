from datetime import datetime
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import sklearn as sk
import pytz
from math import sqrt, cos, radians
from typing import List, Tuple
from geopy.distance import great_circle
from geopy.point import Point

MyPoint = Tuple[float, float, float, datetime] # latitude, longitude, altitude, collection date
StayPoint = Tuple[float, float, datetime, datetime] # latitude, longitude, time of arrival, time of leave

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
TZ = pytz.timezone('GMT')

def convert_row_to_point(row) -> MyPoint:
    converted_date = datetime.strptime(row['time'], DATE_FORMAT)
    gmt_date = TZ.localize(converted_date)
    return (row['lat'], row['lon'], row['alt'], gmt_date)

def distance(p1: MyPoint, p2: MyPoint) -> float:
    # Cannot set altitude, geopy does not support it.
    gp_p1 = Point(latitude=p1[0], longitude=p1[1], altitude=0)
    gp_p2 = Point(latitude=p2[0], longitude=p2[1], altitude=0)

    plain_distance = great_circle(gp_p1, gp_p2).meters
    altitude_delta = (gp_p1.altitude - gp_p2.altitude)

    return sqrt(plain_distance**2 + altitude_delta**2)

def timedelta(p1: MyPoint, p2: MyPoint) -> float:
    return float((p2[3] - p1[3]).seconds)

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

                    stay_point: StayPoint = (
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



def extract_stay_region(trajectories: pd.DataFrame, users, d_thresh: float, t_thresh: float, d: float) -> list:
    """
    trajectories --> phi
    """
    for u_k in users[:1]:
        traj_k = trajectories[trajectories['user'] == u_k]
        S_k: List[StayPoint] = stay_point_detection(traj_k, d_thresh, t_thresh)
        SP.extend(S_k)
        # print("len(traj_k) = ", len(traj_k), ", len(S_k) = ", len(S_k))
    print("created sp")
    #del trajectories
    G_lat, G_lon = grid_division(d)
    print("created grid lat lon")
    # latitude = y, longitude = x
    G_lat_len = len(G_lat)
    G_lon_len = len(G_lon)
    G_lat_i = 0
    newG = {}
    while G_lat_i < (G_lat_len - 1):
        G_lon_i = 0
        while G_lon_i < (G_lon_len - 1):
            # These 4 variables represent the grid square in position (G_lat_i, G_lon_i)
            top_left = G_lat[G_lat_i], G_lon[G_lon_i]
            top_right = G_lat[G_lat_i], G_lon[G_lon_i + 1]
            bottom_left = G_lat[G_lat_i + 1], G_lon[G_lon_i]
            bottom_right = G_lat[G_lat_i + 1], G_lon[G_lon_i + 1]

            newG[str(top_left)+";"+str(top_right)+";"+str(bottom_left)+";"+str(bottom_right)] = {}

            # ecc
            G_lon_i += 1
        G_lat_i += 1

    print("created corner")
    SP_len = len(SP)
    sp_index = 0
    while sp_index < SP_len:
        sp = SP[sp_index]
        # sp[0] #lat
        # sp[1] #lon
        print("start iterating grid")
        for grid,key in newG:
            key_splitted = key.split(";")
            top_left = eval(key_splitted[0])
            top_right = eval(key_splitted[1])
            bottom_left = eval(key_splitted[2])
            bottom_right = eval(key_splitted[3])
            #if a sp is inside the 4 corner it will be added to tha3t grid and removed from SP
            if (sp[0] >= bottom_right[0] and sp[0] <= top_left[0]) and ((sp[1] >= bottom_left[1] and sp[1] <= top_right[1])):
                print("added ",sp," to ",key)
                grid.append(sp)
                SP.remove(sp)
                sp_index -= 1
        sp_index += 1
        print("end iterating grid")

    print("SP len: ", len(SP))
    print("number of grid: ", len(newG))
    print("number of sp in 1st grid: ", len(newG[0]))
    print("sp of first grid:")
    print(newG[0])
        #  maybe index increase/decrease can be removed
        #  because for each stay point we iterate over all grids
        #  so every sp iterated has been assagned to a grid

# SP: List[List[StayPoint]] = list()
SP: List[StayPoint] = list()

def main():
    input_file: str = 'geolife_trajectories_complete.csv'
    d_thresh: float = 100 # Meters
    t_thresh: float = 300 # Seconds
    d: float = 600
    df_trajectories = pd.read_csv(input_file)
    users = np.unique(df_trajectories['user']) # users --> U
    print("lat_max = ", df_trajectories['lat'].max(), "lat_min = ", df_trajectories['lat'].min())
    print("lon_max = ", df_trajectories['lon'].max(), "lon_min = ", df_trajectories['lon'].min())
    # grid_division(d)
    extract_stay_region(df_trajectories, users, d_thresh, t_thresh, d)

if __name__ == '__main__':
    main()
