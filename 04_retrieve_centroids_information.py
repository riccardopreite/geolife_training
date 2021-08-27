import pandas as pd
import requests
import json
# Input file  (output of 03_dbscan_clustering)
INPUT_FILE_CLUSTERED = "03_dbscan_clustering_output/clustered.csv"

# HERE API KEY
API_KEY = "XMJaKLYAY_k6eFJDWS7rXbb9snd7k7p2P5VDRA7o5J0"

REQUEST_URL = "https://discover.search.hereapi.com/v1/discover?limit=1&apiKey="+API_KEY

QUERY_TO_ADD = "&q="
LAT_LON_TO_ADD = "&at="

TYPE_OF_PLACE = ["leisure","restaurant","sport"]

def main():
    df = pd.read_csv(INPUT_FILE_CLUSTERED)
    full_business_json = {}
    for index, row in df.iterrows():
        lat_lon = str(row[1]) + "," + str(row[2])
        # print(lat_lon)
        lat_lon_url = REQUEST_URL + LAT_LON_TO_ADD + lat_lon

        for type in TYPE_OF_PLACE:
            url = lat_lon_url + QUERY_TO_ADD + type
            print("NEW URL")
            print(url)
            r = requests.get(url)
            json_response = json.loads(r.text)
            # print(json_response)
            if len(json_response['items']) != 0:
                business_found = json_response['items'][0]
                print("DISTANCE ", business_found["distance"])
                if business_found["distance"] < 500:
                    full_business_json[lat_lon] = business_found
                else:
                    print("THIS ELEMENT IS OUT OF LIMITS")
                    print(business_found["title"])
            else:
                print("EMPTY RESPONSE")




if __name__ == '__main__':
    main()
