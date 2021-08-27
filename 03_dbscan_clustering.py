from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
import shapefile as shp
import math
# Colors for plotting points onto the map
COLORS: List[str] = ['blue','green','yellow','orange','pink','gray','brown','olive','indigo','springgreen','goldenrod','lightcoral']

# Input file (output of 02_extract_stay_regions)
INPUT_FILE_STAYPOINTS: str = 'output_stay_points.csv'

# Output file (output of 03_dbscan_clustering)
OUTPUT_FILE_STAYPOINTS: str = 'clustered.csv'

# Output file of centroids (output of 03_dbscan_clustering)
OUTPUT_FILE_CENTROIDS: str = 'centroids.csv'

# Map file name (both .shp and .dbf)
MAP_FILE_NAME: str = "land_limits" # "country_limits"

# Distance threshold between points (to be considered in the same cluster)
DISTANCE_THRESHOLD = 100


def plot_map():
    map_shp_file = open(MAP_FILE_NAME + '.shp', 'rb')
    map_dbf_file = open(MAP_FILE_NAME + '.dbf', 'rb')
    map_file = shp.Reader(shp=map_shp_file, dbf=map_dbf_file)
    plt.figure()
    for shape in map_file.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y)


def plot_points_and_map(clusters: np.ndarray, X: np.array, bounds: Tuple[float, float, float, float]):
    plot_map()

    u_cluster_labels: np.ndarray = np.unique(clusters)
    tot_num_clusters = len(u_cluster_labels) - (1 if -1 in clusters else 0)

    print("Found", tot_num_clusters, "clusters.")
    plt.scatter(X[clusters == -1, 1], X[clusters == -1, 0], s = 10, color = 'black')
    for label in u_cluster_labels[1:]:
        # Calculating centroid for the cluster
        centroid = np.mean(X[clusters == label , :], axis=0)
        #drawing centroid
        plt.scatter(centroid[1], centroid[0], s = 10, color = 'red')

    plt.xlim(bounds[0], bounds[1])
    plt.ylim(bounds[2], bounds[3])
    plt.title(f"Points: {len(X)} - Clusters: {tot_num_clusters} - Outliers: {len(X[clusters == -1])}")
    plt.legend()
    plt.show()

def extract_centroid(df: pd.DataFrame ,labels: np.ndarray):
    unique_labels = np.unique(labels)
    centroid_list = list()
    df = np.array(df)
    for label in unique_labels:
        new_centroid = np.mean(df[labels == label , :], axis=0)
        centroid_list.append(new_centroid)

    centroid_df = np.stack(centroid_list)
    centroid_df = pd.DataFrame(centroid_df,columns=["latitude","longitude"])
    centroid_df["cluster"] = unique_labels.tolist()

    centroid_df.to_csv(OUTPUT_FILE_CENTROIDS,index=False)


def main():
    stay_points = pd.read_csv(INPUT_FILE_STAYPOINTS)
    X = pd.DataFrame(stay_points.drop(["time_of_arrival", "time_of_leave", "region_id", "user"], axis=1))
    eps = (DISTANCE_THRESHOLD * 0.001) / 111
    radians_array = np.radians(X)
    radians_eps = math.radians(eps)
    cluster_labels = DBSCAN(eps=radians_eps, min_samples=3,metric='haversine',algorithm='ball_tree').fit_predict(radians_array)

    extract_centroid(X,cluster_labels)

    cluster_to_csv = pd.DataFrame(cluster_labels)
    to_save = pd.concat([X,cluster_to_csv],axis=1)

    to_save.to_csv(OUTPUT_FILE_STAYPOINTS)
    X = np.array(X)

    # Map bounds
    lon_min = -180.0
    lon_max = 180.0
    lat_min = -90.0
    lat_max = 90.0


    bounds = (lon_min, lon_max, lat_min, lat_max)

    print(f"Silhouette Coefficient (should be in [-1,+1]): {metrics.silhouette_score(X, cluster_labels)}")

    plot_points_and_map(cluster_labels, X, bounds)


if __name__ == '__main__':
    main()
