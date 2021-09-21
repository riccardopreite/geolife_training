import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import pickle
from sklearn import tree
import pydot
from six import StringIO
from imblearn.over_sampling import RandomOverSampler

# Input file (output of 08_create_HAR_hour)
INPUT_FILE_NEAREST = "08_create_HAR_hour/create_HAR_hour.csv"

# Output file (output of 09_decision_tree)
OUTPUT_DECISION_TREE = "09_decision_tree/finalized_model_tree"

# Output pdf file (output of 09_decision_tree)
OUTPUT_DECISION_TREE_PDF = "09_decision_tree/decisiontree.pdf"


# Columns of result
# cols_y = ["place_category","place_type"]
cols_y = ["place_category"]
# "time_of_day","day_of_week","har","category_id" will remain
cols_to_drop = ["lat_centroid","lon_centroid","cluster_index","place_name","google_place_id","place_address","place_category","place_type","place_lat","place_lon","distance_to_centroid","time_of_arrival","category_id"]
# cols_to_stay = ["time_of_day","day_of_week","har"]


# Classifier
CLASSIFIER = DecisionTreeClassifier(criterion="entropy",splitter="random",random_state=17)

def train_tree(nearest_point_df):
    # print(cols_to_drop)
    y = nearest_point_df['category_id']
    X = nearest_point_df.drop(cols_to_drop,axis=1)
    

    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    

    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, train_size=0.70,test_size=0.30)
    decision_tree = CLASSIFIER.fit(X_train, y_train)

    y_pred = CLASSIFIER.predict(X_test)
    return decision_tree,y_pred,y_test

def save_decision_tree():
    dateTimeObj = datetime.now()
    date = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day) + '-' + str(dateTimeObj.hour) + '-' + str(dateTimeObj.minute) + '-' + str(dateTimeObj.second)
    filename = OUTPUT_DECISION_TREE + date +'.sav'
    pickle.dump(CLASSIFIER, open(filename, 'wb'))

def save_printed_decision_tree(decision_tree):
    dot_data = StringIO()
    list = ("Hours","Movement","Day")
    classList = ("Restaurants","Leisure","Sport")
    tree.export_graphviz(decision_tree, out_file=dot_data,feature_names=list,class_names=classList)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    graph[0].write_pdf(OUTPUT_DECISION_TREE_PDF)  # must access graph's first element


def main():
    nearest_point = pd.read_csv(INPUT_FILE_NEAREST)

    # do dummy encoding

    decision_tree, y_pred,y_test = train_tree(nearest_point)
    score = accuracy_score(y_test,y_pred)
    right = accuracy_score(y_test,y_pred,normalize=False)
    # save_decision_tree()

    save_printed_decision_tree(decision_tree)
    print("Score IS ",score," right sample ", right)

if __name__ == '__main__':
    main()