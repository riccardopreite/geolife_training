import pandas as pd
import numpy as np
from os import environ, path
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

# Categories

RESTAURANT_TYPES = ["bakery", "bar", "cafe", "drugstore", "liquor_store", "meal_delivery", "meal_takeway", "restaurant"]

LEISURE_TYPES = ["amusement_park", "aquarium", "art_gallery", "bowling_alley", "casino", "movie_theater", "museum", "night_club", "park", "shopping_mall", "spa", "stadium", "tourist_attraction", "zoo"]

SPORT_TYPES = ["gym"]

PLACE_CATEGORIES = {
    "restaurants" : RESTAURANT_TYPES,
    "leisure" : LEISURE_TYPES,
    "sport" : SPORT_TYPES
}

# Input file  (output of 03_dbscan_clustering)
INPUT_FILE_CLUSTERED = "03_dbscan_clustering_output/centroids.csv"

# Output file (output of 04_retrieve_centroids_information_gmaps)
OUTPUT_FILE_POINTS_API = "04_retrieve_centroids_information_gmaps/points.csv"

# Google Maps cached API Responses
CACHE_GMAPS_RESPONSES = "04_retrieve_centroids_information_gmaps/gmaps_api_responses.json"

# Radius for searching restaurants, leisure and sport places from the points.
SEARCH_RADIUS = 500

# 9 Columns name in the resulting DataFrame
COLUMNS = ["lat_centroid", "lon_centroid", "cluster_index", "place_name", "google_place_id", "place_address", "place_category", "place_type", "place_lat", "place_lon", "distance_to_centroid"]

# Cluster index representing outliers.
OUTLIER_CLUSTER = -1

def main():
    gmaps_api_key = environ.get("GMAPS_API_KEY", default="")
    gmaps = googlemaps.Client(key=gmaps_api_key)
    gmaps_cache = {}
    if path.exists(CACHE_GMAPS_RESPONSES):
        with open(CACHE_GMAPS_RESPONSES, mode="r", encoding="utf8") as cache_file:
            gmaps_cache = json.loads(cache_file.read())
        with open(CACHE_GMAPS_RESPONSES + ".bak", mode="w+", encoding="utf8") as backup_file:
            backup_file.write(json.dumps(gmaps_cache))

    centroids_df = pd.read_csv(INPUT_FILE_CLUSTERED)
    gmaps_api_responses = {}

    points = list()

    max_centroid_index = len(centroids_df) - 1
    for index, centroid in centroids_df.iterrows():
        print(f"@ Searching points for point {index}/{max_centroid_index} @")
        # Rewriting result_json inside gmaps_api_responses every 5 centroids.
        if index % 5 == 0:
            print(f"---> Saving Google Maps API responses in the cache file ({CACHE_GMAPS_RESPONSES}).")
            with open(CACHE_GMAPS_RESPONSES, "w+", encoding="utf8") as file:
                result_json = json.dumps(gmaps_api_responses, indent=2)
                file.write(result_json)

        if centroid[2] == OUTLIER_CLUSTER:
            print("---> This point is an outlier: SKIPPED.")
            continue

        if index > 100:
            break
        
        lat_lon = centroid[0], centroid[1]
        centroid_key = str(centroid[0]) + "," + str(centroid[1])
        gmaps_api_responses[centroid_key] = {}
        p_start = Point(lat_lon)

        for category_key, category in PLACE_CATEGORIES.items():
            print(f"-> Category {category_key}")
            # equator
            gmaps_api_responses[centroid_key][category_key] = {}
            min_distance = 1000
            stored_result = ()
            for place_type in category:
                print(f"--> Place type {place_type}")
                gmaps_api_responses[centroid_key][category_key][place_type] = {}

                if centroid_key in gmaps_cache.keys() and \
                    category_key in gmaps_cache[centroid_key].keys() and \
                    place_type in gmaps_cache[centroid_key][category_key].keys():
                    print(f"---> Places of place type {place_type}, category {category_key} of centroid {centroid_key} were found in the cache file.")
                    # Saving in a variable called json_response to mimick the else branch.
                    json_response = {} # gmaps_cache[centroid_key][category_key][place_type]
                    json_response['results'] = list()
                    cached_place = next(iter(gmaps_cache[centroid_key][category_key][place_type]), "")
                    if cached_place != "":
                        cached_place = gmaps_cache[centroid_key][category_key][place_type][cached_place]["results"][0]
                        if "geometry" in cached_place.keys() and "location" in cached_place["geometry"].keys():
                            cached_place_loc = cached_place["geometry"]["location"]
                        else:
                            cached_place_loc = {}
                        json_response['results'].append({
                            "name": cached_place["name"] if "name" in cached_place.keys() else "no-name",
                            "place_id": cached_place["place_id"] if "place_id" in cached_place.keys() else "no-place-id",
                            "vicinity": cached_place["vicinity"] if "vicinity" in cached_place.keys() else "vicinity",
                            "geometry": {
                                "location": {
                                    "lat": cached_place_loc["lat"] if "lat" in cached_place_loc.keys() else "no-lat",
                                    "lng": cached_place_loc["lng"] if "lng" in cached_place_loc.keys() else "no-lng"
                                }
                            }
                        })
                else:
                    json_response = gmaps.places_nearby(
                        location = lat_lon,
                        radius = SEARCH_RADIUS,
                        language = "en-GB",
                        type = place_type
                    )

                if len(json_response['results']) > 0:
                    res_0 = json_response["results"][0]
                    name = res_0["name"] if "name" in res_0.keys() else "no-name"
                    place_id = res_0["place_id"] if "place_id" in res_0.keys() else "no-place-id"
                    address = res_0["vicinity"] if "vicinity" in res_0.keys() else "no-vicinity"
                    if "geometry" in res_0.keys() and "location" in res_0["geometry"].keys():
                        res_0_loc = res_0["geometry"]["location"]
                        lat_resulted = res_0_loc["lat"] if "lat" in res_0_loc.keys() else "no-lat"
                        lon_resulted = res_0_loc["lng"] if "lng" in res_0_loc.keys() else "no-lng"
                    
                    p_arrive = Point(lat_resulted,lon_resulted)

                    gmaps_api_responses[centroid_key][category_key][place_type][str(lat_resulted)+","+str(lon_resulted)] = json_response
                    
                    distance_between_points = distance(p_start,p_arrive)

                    if distance_between_points <= min_distance:
                        min_distance = distance_between_points
                        stored_result = (
                            centroid[0],
                            centroid[1],
                            centroid[2],
                            name,
                            place_id,
                            address,
                            category_key,
                            place_type,
                            lat_resulted,
                            lon_resulted,
                            distance_between_points
                        )

            if len(stored_result) > 0:
                points.append(stored_result)
            # END for place_type in category
        # END for category_key, category in PLACE_CATEGORIES.items()
    # END for index, centroid in centroids_df.iterrows()

    # Exporting gmaps_api_responses to JSON.
    print(f"---> Saving Google Maps API responses in the cache file ({CACHE_GMAPS_RESPONSES}).")
    with open(CACHE_GMAPS_RESPONSES, "w+", encoding="utf8") as file:
        result_json = json.dumps(gmaps_api_responses, indent=2)
        file.write(result_json)

    # Exporting points to CSV.
    print(f"---> Saving to CSV the collected data inside {OUTPUT_FILE_POINTS_API}.")
    points_numpyzed = np.stack(points)
    points_df = pd.DataFrame(points_numpyzed, columns=COLUMNS)

    points_df.to_csv(OUTPUT_FILE_POINTS_API, index=False)
    # Workaround per evitare che la colonna cluster_index sia di float.
    points_df = pd.read_csv(OUTPUT_FILE_POINTS_API, dtype={'cluster_index': int})
    points_df.to_csv(OUTPUT_FILE_POINTS_API, index=False)


if __name__ == '__main__':
    main()
