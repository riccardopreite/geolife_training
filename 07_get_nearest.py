import pandas as pd

# Input file (output of 06_convert_timezone)
INPUT_FILE_POINTS_WITH_TZ = "06_convert_timezone/points_with_toa_tz.csv"

# Output file (output of 07_get_nearest)
OUTPUT_FILE_NEAREST = "07_get_nearest/nearest_points.csv"

# Columns name from INPUT_FILE
cols = ["lat_centroid","lon_centroid","cluster_index","place_name","google_place_id","place_address","place_category","place_type","place_lat","place_lon","distance_to_centroid","time_of_arrival"]

def main():
    points_with_toa = pd.read_csv(INPUT_FILE_POINTS_WITH_TZ)

    nearest_point = pd.DataFrame(columns=cols)
    clusters = points_with_toa["cluster_index"].unique()

    for cluster in clusters:
        same_centroid_sp = points_with_toa[points_with_toa['cluster_index'] == cluster]

        min_index = same_centroid_sp["distance_to_centroid"].idxmin(axis=1)

        nearest_point = nearest_point.append(points_with_toa.loc[min_index],ignore_index=False)

    nearest_point.to_csv(OUTPUT_FILE_NEAREST,index=False)

if __name__ == '__main__':
    main()