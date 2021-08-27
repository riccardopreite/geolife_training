import pandas as pd
from os import environ
import googlemaps
import json

# timezonefinder

# ["results"][0]["name"]
# ["results"][0]["types"]
# ["results"][0]["place_id"]
# ["results"][0]["vicinity"]
# ["results"][0]["geometry"]["location"]["lat"]
# ["results"][0]["geometry"]["location"]["lng"]



# Input file  (output of 03_dbscan_clustering)
INPUT_FILE_CLUSTERED = "03_dbscan_clustering_output/clustered.csv"

# Radius for searching restaurants, leisure and sport places from the points.
SEARCH_RADIUS = 500

def main():
    # points = pd.read_csv(INPUT_FILE_CLUSTERED)
    gmaps_api_key = environ.get("GMAPS_API_KEY", default="")
    gmaps = googlemaps.Client(key=gmaps_api_key)

    diso_coord = 40.00872520653512,18.388315657587032
    result = gmaps.places_nearby(
            location = diso_coord,
            radius = SEARCH_RADIUS,
            language = "en-GB",
            type = "gas_station"
    )
    # print(result)

    # for index, row in points.iterrows():
    #     print(index, row)
    #     result = gmaps.places_nearby(
    #         location = (row['latitude'], row['longitude']),
    #         radius = SEARCH_RADIUS,
    #         language = "en-GB",
    #         type = "restaurant"
    #     )
    with open("places_api.json", "w+", encoding="utf8") as file:
        result_json = json.dumps(result)
        file.write(result_json)
    # break
        


if __name__ == '__main__':
    main()
