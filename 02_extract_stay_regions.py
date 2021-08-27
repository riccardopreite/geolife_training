from datetime import datetime
import pandas as pd
import numpy as np
import pytz
from math import sqrt
from typing import Dict, List, Tuple
from geopy.distance import great_circle
from geopy.point import Point
from os.path import exists as exists_file

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
    def __init__(self, latitude: float, longitude: float, time_of_arrival: datetime, time_of_leave: datetime, user: int) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.time_of_arrival = time_of_arrival
        self.time_of_leave = time_of_leave
        self.region_id = -2
        self.user = user

    def to_tuple(self) -> StayPointTuple:
        return self.latitude, self.longitude, self.time_of_arrival, self.time_of_leave, self.region_id

    def to_csv_row(self) -> str:
        return str(self.latitude) + "," + str(self.longitude) + "," + self.time_of_arrival.isoformat() + "," + self.time_of_leave.isoformat() + "," + str(self.region_id) + "," + str(self.user) + "\n"


class Region(object):
    def __init__(self, region_id: int, latitude: float, longitude: float) -> None:
        super().__init__()
        self.region_id = region_id
        self.latitude = latitude
        self.longitude = longitude

    def to_tuple(self) -> RegionTuple:
        return self.region_id, self.latitude, self.longitude
    
    def to_csv_row(self) -> str:
        return str(self.region_id) + "," + str(self.latitude) + "," + str(self.longitude) + "\n"


def convert_row_to_point(row) -> MyPoint:
    """
    Takes in input a pd.Dataframe row and returns a MyPoint instance with the localized date.
    """
    converted_date = datetime.strptime(row['time'], DATE_FORMAT)
    gmt_date = TZ.localize(converted_date)
    return MyPoint(row['lat'], row['lon'], row['alt'], gmt_date)


def distance(p1: MyPoint, p2: MyPoint) -> float:
    """
    Computes the approximate distance (in meters) between p1 and p2 and returns it.
    """
    # Cannot set altitude, geopy does not support it.
    gp_p1 = Point(latitude=p1.latitude, longitude=p1.longitude, altitude=0)
    gp_p2 = Point(latitude=p2.latitude, longitude=p2.longitude, altitude=0)

    plain_distance = great_circle(gp_p1, gp_p2).meters
    altitude_delta = (gp_p1.altitude - gp_p2.altitude)

    return sqrt(plain_distance**2 + altitude_delta**2)


def timedelta(p1: MyPoint, p2: MyPoint) -> float:
    """
    Computes the time difference between the collection_date of p1 and p2.
    """
    return float((p2.collection_date - p1.collection_date).seconds)


def get_centroid_coordinates(grid_cell_in_region_center_key: str) -> Tuple[float, float]:
    """
    Returns the centroid coordinates for the given corner coordinates (encoded as a string, the argument of the function).
    """
    corner_coordinates: List[Tuple[float,float]] = list(map(lambda coord_couple: eval(coord_couple), grid_cell_in_region_center_key.split(";")))
    top_left = corner_coordinates[0]
    top_right = corner_coordinates[1]
    bottom_left = corner_coordinates[2]
    bottom_right = corner_coordinates[3]
    center_lat, center_lon = np.average([
        top_left[0], top_right[0], bottom_left[0], bottom_right[0]
    ]), np.average([
        top_left[1], top_right[1], bottom_left[1], bottom_right[1]
    ])
    
    return center_lat, center_lon


def get_neighbor_grids(g: str, G: Dict[str, List[int]], d: float) -> List[str]:
    """
    g is the key for accessing the grid cell for which neighboring cells must be found.
    G is the whole grid.
    d is the grid cell side (needed for finding neighbors).
    """
    corner_coordinates: List[Tuple[float,float]] = list(map(lambda coord_couple: eval(coord_couple), g.split(";")))
    top_left = corner_coordinates[0]

    gc_top_left = (top_left[0] + d, top_left[1] - d)
    gc_bottom_left = (top_left[0], top_left[1] - d)
    gc_top_right = (top_left[0] + d, top_left[1])
    gc_bottom_right = (top_left[0], top_left[1])

    neighbors_key_list: List[str] = list()

    for i in range(3):
        for j in range(3):
            top_left = tuple(gc_top_left[0] - (d * i), gc_top_left[1] + (d * j))
            top_right = tuple(gc_top_right[0] - (d * i), gc_top_right[1] + (d * j))
            bottom_left = tuple(gc_bottom_left[0] - (d * i), gc_bottom_left[1] + (d * j))
            bottom_right = tuple(gc_bottom_right[0] - (d * i), gc_bottom_right[1] + (d * j))
            key_in_G = str(top_left) + ";" + str(top_right) + ";" + str(bottom_left) + ";" + str(bottom_right)
            if key_in_G in G: # Iterates over G.keys()
                neighbors_key_list.append(key_in_G)
    
    return neighbors_key_list


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
        +delta
    ) #+delta
    for lon_range in longitude_range_unique[1:]:
        new_lon_grid = np.arange(
            lon_range[0],
            lon_range[1],
            +delta
        )
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
                    if (sp.latitude >= bottom_left[0] and sp.latitude <= top_left[0]) and (sp.longitude >= bottom_left[1] and sp.longitude <= bottom_right[1]):
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


def stay_point_detection(traj_k: pd.DataFrame, d_thresh: float, t_thresh: float, user: int) -> List[StayPoint]:
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
                        np.average([cp.latitude for cp in current_points]),
                        np.average([cp.longitude for cp in current_points]),
                        current_points[0].collection_date,
                        current_points[-1].collection_date,
                        user
                    )
                    stay_points.append(stay_point)
                    i = j - 1
                keep = False
        i += 1
    return stay_points


def assign_region(G: Dict[str, List[int]], d: float) -> List[Region]:
    unassigned_stay_points_count: List[Tuple[str, int]] = list()
    for G_key, G_val in G.items():
        unassigned_stay_points_count.append(tuple(G_key, len(G_val)))
    unassigned_stay_points_count.sort(key=lambda t: t[1], reverse=True)

    regions: List[Region] = list()

    while len(unassigned_stay_points_count) > 0:
        """i is the key of the grid cell containing the maximum number of stay points."""
        i = unassigned_stay_points_count[0][0]
        center_lat, center_lon = get_centroid_coordinates(i)
        
        """
        r contains at most 9 items which could possibly be keys for accessing stay point lists in G.
        """
        r = get_neighbor_grids(i, G, d) # r = G[i] U ng
        for region in r:
            unassigned_stay_points_count.remove(region)
            for sp_index in G[region]:
                SP[sp_index].region_id = len(regions)
        
        regions.append(Region(len(regions), center_lat, center_lon))

        unassigned_stay_points_count.sort(key=lambda t: t[1], reverse=True)

    return regions    


def extract_stay_region(input_file_template: str, users: int, d_thresh: float, t_thresh: float, d: float) -> List[Region]:
    print("[1/4] Generating Stay Points for users")
    
    for u_k in range(users):
        if u_k in USERS_TO_SKIP:
            print("- [", u_k + 1, "/", users, "] Skipped")
            continue
        trajectories_k = pd.read_csv(input_file_template.replace("x", str(u_k)))
        S_k: List[StayPoint] = stay_point_detection(trajectories_k, d_thresh, t_thresh, u_k)
        SP.extend(S_k)
        print("- [", u_k + 1, "/", users, "] len(trajectories_k) = ", len(trajectories_k), ", len(S_k) = ", len(S_k), ", len(SP) = ", len(SP))
        with open(OUTPUT_FILE_STAYPOINTS, mode="a", encoding="utf8") as file:
            file.writelines([sp.to_csv_row() for sp in S_k])
    
    #del trajectories
    print("[2/4] Generating the grid axes")
    G_lat, G_lon = grid_division(d)
    # latitude = y, longitude = x

    print("[3/4] Assigning Stay Points to grid cells")
    G = assign_stay_points_to_grid_cell(G_lat, G_lon)
    print("number of grid cells: ", len(G))
    print("number of sp in 1st grid: ", len(G[0]))
    print("sp of first grid:")
    print(G[0])

    print("[4/4] Assigning Stay Points to regions")
    regions = assign_region(G, d)

    return regions


# List of generated stay points.
SP: List[StayPoint] = list()

# List of regions
R = list()

# Distance threshold in meters
DISTANCE_THRESHOLD: float = 100

# Time threshold in seconds
TIME_THRESHOLD: float = 300

# Length of a grid cell side in meters
GRID_SIDE_LENGTH: float = 600

# Total number of users
USERS_CARDINALITY: int = 182

# Users to skip from analysis
USERS_TO_SKIP: List[int] = list()

# Input file (x must be replaced with the user number)
INPUT_FILE_TEMPLATE_NAME: str = '01_read_geo_output/user_data/geolife_geolife_trajectories_user_x.csv'

# Output file for stay points
OUTPUT_FILE_STAYPOINTS: str = '02_extract_stay_regions_output/output_stay_points.csv'

# Output file for regions
OUTPUT_FILE_REGIONS: str = '02_extract_stay_regions_output/output_stay_regions.csv'

def main():
    if not exists_file(OUTPUT_FILE_STAYPOINTS):
        with open(OUTPUT_FILE_STAYPOINTS, mode="w+", encoding="utf8") as file:
            file.write("latitude,longitude,time_of_arrival,time_of_leave,region_id,user\n")

    if not exists_file(OUTPUT_FILE_REGIONS):
        with open(OUTPUT_FILE_REGIONS, mode="w+", encoding="utf8") as file:
            file.write("region_id,latitude,longitude\n")

    R = extract_stay_region(INPUT_FILE_TEMPLATE_NAME, USERS_CARDINALITY, DISTANCE_THRESHOLD, TIME_THRESHOLD, GRID_SIDE_LENGTH)
    
    with open(OUTPUT_FILE_REGIONS, mode="w+", encoding="utf8") as file:
        file.write("region_id,latitude,longitude\n")
        file.writelines([r.to_csv_row() for r in R])

if __name__ == '__main__':
    main()
