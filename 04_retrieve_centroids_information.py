import pandas as pd
import requests
import json
# Input file  (output of 03_dbscan_clustering)
INPUT_FILE_CLUSTERED = "clustered.csv"

# HERE API KEY
API_KEY = "yBQMRCYwyWM7QB3MFT6dAjHu0HGFb_lEpvWQava8TNo"

REQUEST_URL = "https://discover.search.hereapi.com/v1/discover?limit=1&apiKey="+API_KEY

QUERY_TO_ADD = "&q="
LAT_LON_TO_ADD = "&at="

TYPE_OF_PLACE = ["luisire","restaurant","sport"]

def main():
    df = pd.read_csv(INPUT_FILE_CLUSTERED)

    for index, row in df.iterrows():
        lat_lon = str(row[1]) + "," + str(row[2])
        # print(lat_lon)
        lat_lon_url = REQUEST_URL + LAT_LON_TO_ADD + lat_lon

        for type in TYPE_OF_PLACE:
            url = lat_lon_url + QUERY_TO_ADD + type
            # print(url)
            r = requests.get(url)
            json_response = json.loads(r.text)
            print(json_response)

        if index == 0:
            return


if __name__ == '__main__':
    main()
