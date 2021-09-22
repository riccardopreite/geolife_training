from datetime import datetime
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

def convert_time_zone(new_timezone_str: str, old_datetime: datetime) -> datetime:
    """
    Returns old_datetime shifted in the new timezone.
    """
    new_timezone = pytz.timezone(new_timezone_str)

    # two-step process (first localize the date, then convert it).
    localized_timestamp = OLD_TIMEZONE.localize(old_datetime)
    new_timezone_timestamp = localized_timestamp.astimezone(new_timezone)
    return new_timezone_timestamp

def get_timezone(latitude: float, longitude: float) -> str:
    """
    Returns the timezone of a place given its coordinates.
    """

    tf = TimezoneFinder()
    new_timezone = tf.timezone_at(lng=longitude, lat=latitude)
    return new_timezone

def main():    
    points_with_toa = pd.read_csv(INPUT_FILE_POINTS_WITH_TOA)

    points_with_toa_tz = points_with_toa.copy()

    # sp_toa = stay point with time of arrival
    for index, sp_toa in points_with_toa.iterrows():
        # Get new TimeZone based on the point's location.
        lat, lon = sp_toa["place_lat"],sp_toa["place_lon"]
        new_timezone = get_timezone(latitude=lat, longitude=lon)

        # Convert datetime to new TimeZone.
        old_toa_str: str = sp_toa["time_of_arrival"]
        old_toa_datetime = datetime.strptime(old_toa_str,'%Y-%m-%d %H:%M:%S')
        new_datetime = convert_time_zone(new_timezone, old_toa_datetime)

        # Update dataframe cell value.
        points_with_toa_tz.at[index, "time_of_arrival"] = new_datetime
        

    points_with_toa_tz.to_csv(OUTPUT_FILE_POINTS_WITH_TZ, index=False)

if __name__ == '__main__':
    main()