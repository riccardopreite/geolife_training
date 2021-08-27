import pandas as pd
from os import environ
import googlemaps
import json

# Input file  (output of 03_dbscan_clustering)
INPUT_FILE_CLUSTERED = "clustered.csv"

# Radius for searching restaurants, leisure and sport places from the points.
SEARCH_RADIUS = 500

def main():
    points = pd.read_csv(INPUT_FILE_CLUSTERED)
    gmaps_api_key = environ.get("GMAPS_API_KEY", default="")
    gmaps = googlemaps.Client(key=gmaps_api_key)

    for index, row in points.iterrows():
        print(index, row)
        result = gmaps.places_nearby(
            location = (row['latitude'], row['longitude']),
            radius = SEARCH_RADIUS,
            language = "en-GB",
            type = "restaurant"
        )
        with open("places_api.json", "w+", encoding="utf8") as file:
            result_json = json.dumps(result)
            file.write(result_json)
        break
        


if __name__ == '__main__':
    main()
