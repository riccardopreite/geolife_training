import pandas as pd 
import datetime
from random import randint
import numpy as np
##########

#  HAR
# 1: Bike
# 2: Car/Bus
# 3: Walk
# 4: Still

##########
# Input file (output of 07_get_nearest)
INPUT_FILE_NEAREST = "07_get_nearest/nearest_points.csv"
INPUT_FILE_POINTS_WITH_TZ = "06_convert_timezone/points_with_toa_tz.csv"

# Output file (output of 08_create_HAR_hour)
OUTPUT_FILE_HAR_TIME = "08_create_HAR_hour/create_HAR_hour.csv"
OUTPUT_FILE_HAR_TIME_NOT_NEAREST = "08_create_HAR_hour/create_HAR_hour_not_nearest.csv"

# New columns of nearest point
cols = ["lat_centroid","lon_centroid","cluster_index","place_name","google_place_id","place_address","place_category","place_type","place_lat","place_lon","distance_to_centroid","time_of_arrival","time_of_day","day_of_week","har","category_id"]

def get_point(input_file,output_file):
    nearest_point = pd.read_csv(input_file)
    har_point = pd.DataFrame([],columns=cols)
    categories_unique = nearest_point["place_category"].unique().tolist()
    for _, n_sp in nearest_point.iterrows():

        date = datetime.datetime.strptime(n_sp["time_of_arrival"],'%Y-%m-%d %H:%M:%S%z')
        seconds_time = (date.hour * 3600) + (date.minute * 60) + date.second
        # 0 = Monday, 6 = Sunday
        day_of_week = date.weekday()

        # .index(elem) = .indexOf(elem)
        category_id = categories_unique.index(n_sp["place_category"])
        if n_sp["place_category"] == "restaurant":
            har = randint(3, 4)
        else:
            har = randint(1, 4)


        # make first dataset with string and encoding them in 09.py
        n_sp["time_of_day"] = seconds_time
        n_sp["day_of_week"] = day_of_week
        n_sp["har"] = har
        n_sp["category_id"] = category_id


        har_point = har_point.append(n_sp,ignore_index=False)

    har_point.to_csv(output_file,index=False)


def main():
    get_point(INPUT_FILE_NEAREST,OUTPUT_FILE_HAR_TIME)
    # print("FINE PRIMO GET POINT")
    # get_point(INPUT_FILE_POINTS_WITH_TZ,OUTPUT_FILE_HAR_TIME_NOT_NEAREST)

    

    
if __name__ == '__main__':
    main()