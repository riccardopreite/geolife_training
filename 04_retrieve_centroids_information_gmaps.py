import pandas as pd
import numpy as np
from os import environ
import googlemaps
import json
from geopy.distance import great_circle
from geopy.point import Point
from math import sqrt


def distance(p1: Point, p2: Point) -> float:
    """
    Computes the approximate distance (in meters) between p1 and p2 and returns it.
    """
    plain_distance = great_circle(p1, p2).meters

    return plain_distance


# timezonefinder

# ["results"][0]["name"]
# ["results"][0]["types"]
# ["results"][0]["place_id"]
# ["results"][0]["vicinity"]
# ["results"][0]["geometry"]["location"]["lat"]
# ["results"][0]["geometry"]["location"]["lng"]

RESTAURANTS = ["bakery", "bar", "cafe", "drugstore", "liquor_store", "meal_delivery", "meal_takeway", "restaurant"]

LEISURE = ["amusement_park", "aquarium", "art_gallery", "bowling_alley", "casino", "movie_theater", "museum", "night_club", "park", "shopping_mall", "spa", "stadium", "tourist_attraction", "zoo"]

SPORT = ["gym"]

TYPE_OF_POINT = {
    "restaurants" : RESTAURANTS,
    "leisure" : LEISURE,
    "sport" : SPORT
}


# Input file  (output of 03_dbscan_clustering)
INPUT_FILE_CLUSTERED = "03_dbscan_clustering_output/centroids.csv"

# Output file  (output of 04_retrieve_centroids_information_gmaps)
OUTPUT_FILE_POINTS_API = "04_retrieve_centroids_information_gmaps/points.csv"

# Radius for searching restaurants, leisure and sport places from the points.
SEARCH_RADIUS = 500

# 9 Columns name for resulting DataFrame

COLUMNS = ["lat_centroid", "lon_centroid", "cluster_index", "place_name", "google_place_id", "place_address", "place_lat", "place_lon", "distance_to_centroid"]



def main():
    # points = pd.read_csv(INPUT_FILE_CLUSTERED)
    gmaps_api_key = environ.get("GMAPS_API_KEY", default="")
    gmaps = googlemaps.Client(key=gmaps_api_key)


    clustered_centroids = pd.read_csv(INPUT_FILE_CLUSTERED)
    full_json = {}

    points = list()

    for index, centroid in clustered_centroids.iterrows():
        if centroid[2] == -1:
            continue
        elif index > 5:
            break
        
        lat_lon = centroid[0],centroid[1]
        p_start = Point(lat_lon)

        for sub_type_key,sub_type in TYPE_OF_POINT.items():
            # equator
            full_json[sub_type_key] = {}
            min_distance = 1000
            point_resulted = ()
            for place_type in sub_type:
                full_json[sub_type_key][place_type] = {}

                json_response = gmaps.places_nearby(
                    location = lat_lon,
                    radius = SEARCH_RADIUS,
                    language = "en-GB",
                    type = place_type
                )

                if len(json_response['results']) > 0:
                    
                    name = json_response["results"][0]["name"]
                    place_id = json_response["results"][0]["place_id"]
                    address = json_response["results"][0]["vicinity"]
                    lat_resulted = json_response["results"][0]["geometry"]["location"]["lat"]
                    lon_resulted = json_response["results"][0]["geometry"]["location"]["lng"]
                    
                    p_arrive = Point(lat_resulted,lon_resulted)

                    full_json[sub_type_key][place_type][str(lat_resulted)+","+str(lon_resulted)] = json_response
                    
                    distance_between_points = distance(p_start,p_arrive)

                    if distance_between_points <= min_distance:
                        
                        min_distance = distance_between_points
                        point_resulted = (centroid[0],centroid[1],centroid[2],name,place_id,address,lat_resulted,lon_resulted,distance_between_points)


            if len(point_resulted) > 0:
                points.append(point_resulted)

    stack = np.stack(points)
    points_df = pd.DataFrame(stack,columns=COLUMNS)

    points_df.to_csv(OUTPUT_FILE_POINTS_API,index=False)

    with open("full.json", "w+", encoding="utf8") as file:
        result_json = json.dumps(full_json)
        file.write(result_json)

if __name__ == '__main__':
    main()
