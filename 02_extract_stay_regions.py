from datetime import datetime
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

def meters_to_lat_lon_displacement(m, origin_latitude):
    # TODO verificare se ha senso
    lat = m/111111
    lon = m/(111111 * cos(radians(origin_latitude)))
    return lat, lon

def grid_division(d: float):
    """
    d is the side length of a part of the grid.
    """
    globe_max_coordinates = ((90.0, 180.0), (90.0, -180.0), (-90.0, -180.0), (-90.0, 180.0)) # NE, NW, SW, SE


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
            print(i, j)
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
        SP.append(S_k)
        print("len(traj_k) = ", len(traj_k), ", len(S_k) = ", len(S_k))
    G = grid_division(d)

SP: List[List[StayPoint]] = list()

def main():
    input_file: str = 'geolife_trajectories_1to12.csv'
    d_thresh: float = 100 # Meters
    t_thresh: float = 300 # Seconds
    d: float = 500
    df_trajectories = pd.read_csv(input_file)
    users = np.unique(df_trajectories['user']) # users --> U
    extract_stay_region(df_trajectories, users, d_thresh, t_thresh, d)

if __name__ == '__main__':
    main()