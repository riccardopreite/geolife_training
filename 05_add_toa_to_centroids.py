from datetime import datetime
import pandas as pd
import numpy as np
from os import environ, path

# Categories

# Input file (output of 02_extract_stay_regions_output)
INPUT_FILE_OUTPUT_STAY_POINTS = "02_extract_stay_regions_output/output_stay_points.csv"

# Input file (output of 03_dbscan_clustering_output)
INPUT_FILE_CLUSTERED_STAY_POINTS = "03_dbscan_clustering_output/clustered_stay_points.csv"

# Input file (output of 04_retrieve_centroids_information_gmaps)
INPUT_FILE_POINTS = "04_retrieve_centroids_information_gmaps/points.csv"

# Output file (output of 05_add_toa_to_centroids)
OUTPUT_FILE_POINTS_WITH_TOA = "05_add_toa_to_centroids/points_with_toa.csv"

# Cluster index representing outliers.
OUTLIER_CLUSTER = -1

def main():
    output_stay_points = pd.read_csv(INPUT_FILE_OUTPUT_STAY_POINTS)
    clustered_stay_points = pd.read_csv(INPUT_FILE_CLUSTERED_STAY_POINTS)
    points = pd.read_csv(INPUT_FILE_POINTS)

    # Retrieving the needed columns from output_stay_points
    time_of_arrival = output_stay_points.pop('time_of_arrival')
    user = output_stay_points.pop('user')

    clustered_stay_points_with_toa = clustered_stay_points.copy()

    # Adding the retrieved columns to clustered_stay_points_with_toa (copy of clustered_stay_points)
    clustered_stay_points_with_toa['time_of_arrival'] = time_of_arrival
    clustered_stay_points_with_toa['user'] = user

    # Dropping outlier stay points
    clustered_stay_points_with_toa = clustered_stay_points_with_toa[clustered_stay_points_with_toa['cluster'] != OUTLIER_CLUSTER]

    # Getting all the cluster indexes
    clusters = clustered_stay_points_with_toa['cluster'].unique()

    # Adding a placeholder column to hold the times of arrival in points
    points_with_toa = points.copy()
    points_with_toa['time_of_arrival'] = np.nan

    # Calculating the median time of arrival (without considering outlier times, that's why we use quantile(0.5))
    for cluster in clusters:
        stay_points_toa_in_cluster = clustered_stay_points_with_toa[clustered_stay_points_with_toa['cluster'] == cluster]
        times_of_arrival = stay_points_toa_in_cluster['time_of_arrival']
        times_of_arrival = times_of_arrival.astype('datetime64[ns]').view('int64')
        median_toa = times_of_arrival.quantile(0.5, interpolation='nearest').astype('datetime64[ns]') # numpy type
        median_toa = pd.to_datetime(median_toa) # pandas type
        points_with_toa.loc[points_with_toa['cluster_index'] == cluster, 'time_of_arrival'] = median_toa

    points_with_toa.to_csv(OUTPUT_FILE_POINTS_WITH_TOA, index=False)

if __name__ == '__main__':
    main()
