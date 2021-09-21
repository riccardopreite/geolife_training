import datetime
import pytz
import pandas as pd
from timezonefinder import TimezoneFinder

# Input file (output of 05_add_toa_to_centroids)
INPUT_FILE_POINTS_WITH_TOA = "05_add_toa_to_centroids/points_with_toa.csv"

# Output file (output of 06_convert_timezone)
OUTPUT_FILE_POINTS_WITH_TZ = "06_convert_timezone/points_with_toa_tz.csv"


# Original TZ name from dataset
ORIGINAL_TIMEZONE = "UTC"

# Old TZ from dataset
OLD_TIMEZONE = pytz.timezone(ORIGINAL_TIMEZONE)

def convert_time_zone(new_timezone_str: str, datetime):

    # a timestamp I'd like to convert
    # my_timestamp = datetime.datetime.now()

    # create both timezone objects
    
    new_timezone = pytz.timezone(new_timezone_str)

    # two-step process
    localized_timestamp = OLD_TIMEZONE.localize(datetime)
    new_timezone_timestamp = localized_timestamp.astimezone(new_timezone)
    return new_timezone_timestamp

def get_timezone(lat,lon):
    tf = TimezoneFinder()
    new_timezone = tf.timezone_at(lng=lon, lat=lat)
    return new_timezone

def main():    
    points_with_toa = pd.read_csv(INPUT_FILE_POINTS_WITH_TOA)

    points_with_toa_tz = points_with_toa.copy()


    for index, sp_toa in points_with_toa.iterrows():
        # Get new TimeZone
        lat, lon = sp_toa["place_lat"],sp_toa["place_lon"]
        new_timezone = get_timezone(lat=lat,lon=lon)

        # Convert datetime to new TimeZone
        datetime_sp = sp_toa["time_of_arrival"]
        datetime_sp = datetime.datetime.strptime(datetime_sp,'%Y-%m-%d %H:%M:%S')
        new_datetime = convert_time_zone(new_timezone,datetime_sp)

        # Update cell value
        points_with_toa_tz.at[index,"time_of_arrival"] = new_datetime
        

    points_with_toa_tz.to_csv(OUTPUT_FILE_POINTS_WITH_TZ,index=False)

if __name__ == '__main__':
    main()