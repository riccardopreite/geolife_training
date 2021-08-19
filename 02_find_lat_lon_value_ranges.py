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

def main():
    total_num_users = 182
    max_latitudes = list()
    min_latitudes = list()
    max_longitudes = list()
    min_longitudes = list()
    # Used to identify if some users contain wrong values
    users_to_check = list()

    for i in range(total_num_users):
        df = pd.read_csv(f'geolife_geolife_trajectories_user_{i}.csv')
        max_lat = df['lat'].max()
        min_lat = df['lat'].min()
        max_lon = df['lon'].max()
        min_lon = df['lon'].min()
        max_latitudes.append(max_lat)
        min_latitudes.append(min_lat)
        max_longitudes.append(max_lon)
        min_longitudes.append(min_lon)
        if max_lat > 90 or min_lat < -90 or max_lon > 180 or min_lon < -180:
            users_to_check.append(i)

    print("Max latitude in dataset: ", max(max_latitudes))
    print("Min latitude in dataset: ", min(min_latitudes))
    print("Max longitude in dataset: ", max(max_longitudes))
    print("Min longitude in dataset: ", max(min_longitudes))
    print("Please check users with indexes: ", users_to_check)
        

if __name__ == '__main__':
    main()