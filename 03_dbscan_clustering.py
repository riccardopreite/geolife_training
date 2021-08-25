import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import math


colors = ['blue','green','yellow','orange','pink','gray','brown','olive','indigo','springgreen','goldenrod','lightcoral']
INPUT_FILE_STAYPOINTS: str = 'output_stay_points.csv'

d = 600
eps = 0.5

def main():
    stay_points = pd.read_csv(INPUT_FILE_STAYPOINTS)
    stay_points_dropped = pd.DataFrame(stay_points.drop(["time_of_arrival","time_of_leave","region_id","user"],axis=1))
    # It probably will change data
    scaler = StandardScaler()
    # eps = (d * 0.001) / 111
    print("Delta of neighbors: ", eps)

    X = stay_points_dropped

    # X = scaler.fit_transform(stay_points_dropped)
    cluster = DBSCAN(eps=eps,min_samples=10).fit_predict(X)

    X = np.array(X)

    plot(cluster,X)


def plot(label,df):
    #get unique labels
    u_labels = np.unique(label)
    print(u_labels)
    cluster = len(u_labels) - (1 if -1 in label else 0)
    for i in u_labels:
        if i == -1:
            # Black used for noise.
            col = 'black'
        else:
            col = colors[i]
        #drawing cluster
        plt.scatter(df[label == i , 0] , df[label == i , 1] , s=14,color = col)
        #calculating centroid
        centroid = np.mean(df[label == i , :], axis=0)
        #drawing centroid
        plt.scatter(centroid[0] , centroid[1] , s = 2, color = 'red')

    plt.title("n_cluster: %d" %  cluster)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
      # % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
      # % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
      # % metrics.silhouette_score(X, labels))
