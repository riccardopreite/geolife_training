import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import math
import shapefile as shp


colors = ['blue','green','yellow','orange','pink','gray','brown','olive','indigo','springgreen','goldenrod','lightcoral']
INPUT_FILE_STAYPOINTS: str = 'output_stay_points.csv'

d = 600
eps = 0.5

def main():
    stay_points = pd.read_csv(INPUT_FILE_STAYPOINTS)
    X = pd.DataFrame(stay_points.drop(["time_of_arrival","time_of_leave","region_id","user"],axis=1))
    eps = (d * 0.001) / 111
    print("Delta of neighbors: ", eps)

    # X = stay_points_dropped

    # X = scaler.fit_transform(stay_points_dropped)
    cluster = DBSCAN(eps=eps,min_samples=10).fit_predict(X)

    X = np.array(X)

    lon_min = -180 #int(X[:,1].min() - 1)
    lon_max = 180 #int(X[:,1].max() + 1)
    lat_min = -90 #int(X[:,0].min() - 1)
    lat_max = 90 #int(X[:,0].max() + 1)

    BBox = (lon_min, lon_max, lat_min, lat_max)

    # plotOnMap(cluster,X,BBox)
    plotSHP(cluster,X,BBox)
    # plot(cluster,X,BBox)

def plotSHP(label,df,BBox):
    myshp = open('test.shp', 'rb')
    mydbf = open('test.dbf', 'rb')
    sf = shp.Reader(shp=myshp,dbf=mydbf)
    plt.figure()
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y)

    ruh_m = plt.imread('map.png')
    u_labels = np.unique(label)
    print(u_labels)
    cluster = len(u_labels) - (1 if -1 in label else 0)
    # plt.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
    for i in u_labels:
        if i == -1:
            # Black used for noise.
            col = 'black'
            plt.scatter(df[label == i , 1] , df[label == i , 0] , s=40, color = col)
        else:
            # col = colors[i]
            # col = i
            plt.scatter(df[label == i , 1] , df[label == i , 0] , s=40, label=i)
        #drawing cluster
        #calculating centroid
        centroid = np.mean(df[label == i , :], axis=0)
        #drawing centroid
        plt.scatter(centroid[1] , centroid[0] , s = 2, color = 'red')

    plt.xlim(BBox[0],BBox[1])
    plt.ylim(BBox[2],BBox[3])
    plt.title("n_cluster: %d" %  cluster)
    plt.legend()
    plt.show()

def plot(label,df,BBox):
    #get unique labels
    ruh_m = plt.imread('map.png')
    u_labels = np.unique(label)
    print(u_labels)
    cluster = len(u_labels) - (1 if -1 in label else 0)
    plt.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
    ruh_m = plt.imread('map.png')
    for i in u_labels:
        if i == -1:
            # Black used for noise.
            col = 'black'
            plt.scatter(df[label == i , 1] , df[label == i , 0] , s=40, color = col)

        else:
            # col = colors[i]
            # col = i
            plt.scatter(df[label == i , 1] , df[label == i , 0] , s=40, label=i)
        #drawing cluster
        #calculating centroid
        centroid = np.mean(df[label == i , :], axis=0)
        #drawing centroid
        plt.scatter(centroid[1] , centroid[0] , s = 2, color = 'red')

    plt.xlim(BBox[0],BBox[1])
    plt.ylim(BBox[2],BBox[3])
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
