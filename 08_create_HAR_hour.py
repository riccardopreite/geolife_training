import pandas as pd 
from datetime import datetime
from random import randint


def get_har(category: str) -> str:
    if category == "restaurant":
        har_code = randint(4, 5)
    else:
        har_code = randint(1, 5)

    if har_code == 1:
        return "bike"
    elif har_code == 2:
        return "car"
    elif har_code == 3:
        return "bus"
    elif har_code == 4:
        return "walk"
    elif har_code == 5:
        return "still"
    else:
        return ""


##########
# Input files (output of 06_convert_timezone and 07_get_nearest)
INPUT_FILE_NEAREST_POINTS = "07_get_nearest/nearest_points.csv"
INPUT_FILE_POINTS_WITH_TOA_TZ = "06_convert_timezone/points_with_toa_tz.csv"

# Output file (output of 08_create_HAR_hour)
OUTPUT_FILE_HAR_NEAREST = "08_create_HAR_hour/har_nearest.csv"
OUTPUT_FILE_HAR_NOT_NEAREST = "08_create_HAR_hour/har_not_nearest.csv"


def generate_random_har(input_file: str, output_file: str):
    points = pd.read_csv(input_file)

    """
    Creating a copy of the dataset and adding three more columns:
    - time_of_day (time_of_arrival converted in seconds)
    - day_of_week (time_of_arrival's day converted in week day)
    - har (randomly generated based on get_har)
    """
    har_points = points.copy()
    har_points["time_of_day"] = -1 # Seconds (00:00:00 -> 23:59:59 => 0 -> 86399)
    har_points["day_of_week"] = -1 # 0 = Monday, ..., 6 = Sunday
    har_points["har"] = ""

    for index, point in har_points.iterrows():
        date = datetime.strptime(point["time_of_arrival"], '%Y-%m-%d %H:%M:%S%z')
        seconds_time = (date.hour * 3600) + (date.minute * 60) + date.second
        day_of_week = date.weekday()

        har_points.loc[index, "time_of_day"] = seconds_time
        har_points.loc[index, "day_of_week"] = day_of_week
        har_points.loc[index, "har"] = get_har(point["place_category"])

    if(-1 in har_points["time_of_day"].unique() or -1 in har_points["day_of_week"].unique() or "" in har_points["har"].unique()):
        print("There were problems generating the dataset.")

    har_points.to_csv(output_file, index=False)


def main():
    print(f"Generating random human activities and adding time of day (in seconds) and day of week to {INPUT_FILE_NEAREST_POINTS}")
    generate_random_har(INPUT_FILE_NEAREST_POINTS, OUTPUT_FILE_HAR_NEAREST)
    print(f"Generating random human activities and adding time of day (in seconds) and day of week to {INPUT_FILE_POINTS_WITH_TOA_TZ}")
    generate_random_har(INPUT_FILE_POINTS_WITH_TOA_TZ, OUTPUT_FILE_HAR_NOT_NEAREST)

    
if __name__ == '__main__':
    main()