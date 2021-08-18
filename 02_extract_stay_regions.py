from datetime import datetime
import pandas as pd
import numpy as np
import sklearn as sk
import pytz
from math import sqrt
from typing import List, Tuple
from geopy.distance import great_circle
from geopy.point import Point

MyPoint = Tuple[float, float, float, datetime]
StayPoint = Tuple[float, float, datetime, datetime]

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
TZ = pytz.timezone('GMT')

def convert_row_to_point(row) -> MyPoint:
    converted_date = datetime.strptime(row['time'], DATE_FORMAT)
    gmt_date = TZ.localize(converted_date)
    return (row['lat'], row['lon'], row['alt'], gmt_date)

def distance(p1: MyPoint, p2: MyPoint) -> float:
    gp_p1 = Point(latitude=p1[0], longitude=p1[1], altitude=p1[2])
    gp_p2 = Point(latitude=p2[0], longitude=p2[1], altitude=p2[2])

    plain_distance = great_circle(gp_p1, gp_p2).meters
    altitude_delta = (gp_p1.altitude - gp_p2.altitude)

    return sqrt(plain_distance**2 + altitude_delta**2)

def get_centroid_coordinates(SP, r):
    pass

def get_neighbor_grids(g, G):
    pass

def grid_division(d: float):
    pass

def stay_point_detection(traj_k: pd.DataFrame, d_thresh: float, t_thresh: float) -> List[StayPoint]:
    num_rows = traj_k.size
    i = 0
    while i < num_rows:
        # latitude = y, longitude = x
        row_i = traj_k[i]
        p_i: MyPoint = convert_row_to_point(row_i)
        j = i + 1
        while j < num_rows:
            row_j = traj_k[j]
            p_j: MyPoint = convert_row_to_point(row_j)



def extract_stay_region(trajectories: pd.DataFrame, users, d_thresh: float, t_thresh: float, d: float) -> list:
    """
    trajectories --> phi
    """
    for u_k in users:
        traj_k = trajectories[trajectories['user'] == u_k]
        S_k: List[StayPoint] = stay_point_detection(traj_k, d_thresh, t_thresh)
        SP.append(S_k)

SP: List[List[StayPoint]] = list()

def main():
    input_file: str = 'geolife_trajectories_1to12.csv'
    d_thresh: float = 0
    t_thresh: float = 0
    d: float = 0
    df_trajectories = pd.read_csv(input_file)
    users = np.unique(df_trajectories['user']) # users --> U
    extract_stay_region(df_trajectories, users, d_thresh, t_thresh, d)

if __name__ == '__main__':
    main()