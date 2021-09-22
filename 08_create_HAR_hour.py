import pandas as pd 
from datetime import datetime
from random import randint
import numpy as np
##########

#  HAR
# 1: Bike
# 2: Car/Bus
# 3: Walk
# 4: Still

def get_har(har_code: int) -> str:
    if har_code == 1:
        return "bike"
    elif har_code == 2:
        return "car/bus"
    elif har_code == 3:
        return "walk"
    elif har_code == 4:
        return "still"
    else:
        return ""

##########
# Input files (output of 06_convert_timezone and 07_get_nearest)
INPUT_FILE_NEAREST = "07_get_nearest/nearest_points.csv"
INPUT_FILE_POINTS_WITH_TZ = "06_convert_timezone/points_with_toa_tz.csv"

# Output file (output of 08_create_HAR_hour)
OUTPUT_FILE_HAR_NEAREST = "08_create_HAR_hour/har_nearest.csv"
OUTPUT_FILE_HAR_NOT_NEAREST = "08_create_HAR_hour/har_not_nearest.csv"

# New columns of nearest point
cols = ["lat_centroid","lon_centroid","cluster_index","place_name","google_place_id","place_address","place_category","place_type","place_lat","place_lon","distance_to_centroid","time_of_arrival","time_of_day","day_of_week","har","category_id"]

def generate_random_har(input_file: str, output_file: str):
    nearest_point = pd.read_csv(input_file)
    har_points = pd.DataFrame([], columns=cols)

    for _, n_sp in nearest_point.iterrows():
        date = datetime.strptime(n_sp["time_of_arrival"], '%Y-%m-%d %H:%M:%S%z')
        seconds_time = (date.hour * 3600) + (date.minute * 60) + date.second
        # 0 = Monday, 6 = Sunday
        day_of_week = date.weekday()

        # .index(elem) corresponds to .indexOf(elem) in other languages.
        if n_sp["place_category"] == "restaurant":
            har = randint(3, 4)
        else:
            har = randint(1, 4)

        # make first dataset with string and encoding them in 09.py
        n_sp["time_of_day"] = seconds_time
        n_sp["day_of_week"] = day_of_week
        n_sp["har"] = get_har(har)

        har_points = har_points.append(n_sp, ignore_index=False)

    har_points.to_csv(output_file, index=False)


def main():
    generate_random_har(INPUT_FILE_NEAREST, OUTPUT_FILE_HAR_NEAREST)
    generate_random_har(INPUT_FILE_POINTS_WITH_TZ, OUTPUT_FILE_HAR_NOT_NEAREST)

    
if __name__ == '__main__':
    main()