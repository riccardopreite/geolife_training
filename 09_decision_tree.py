import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import pickle
from sklearn import tree
import pydot
from six import StringIO
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder

# Input file (output of 08_create_HAR_hour)
INPUT_FILE_HAR = "08_create_HAR_hour/har_nearest.csv"

# Output file (output of 09_decision_tree)
OUTPUT_DECISION_TREE = "09_decision_tree/decisiontree.sav"

# Output pdf file (output of 09_decision_tree)
OUTPUT_DECISION_TREE_PDF = "09_decision_tree/decisiontree.pdf"

# Columns of result
cols_to_drop = ["cluster_index","place_name","google_place_id","place_address","place_type","place_lat","place_lon","distance_to_centroid","time_of_arrival"]

# Classifier and oversampler
CLASSIFIER = DecisionTreeClassifier(criterion="entropy", splitter="random", random_state=17)
OVERSAMPLER = RandomOverSampler(sampling_strategy='minority', random_state=17)


def oversample(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Oversamples the X and y and returns the ovesampled versions.
    """
    global OVERSAMPLER

    # fit and apply the transformation for oversampling
    X_oversampled, y_oversampled = OVERSAMPLER.fit_resample(X, y)

    return X_oversampled, y_oversampled


def train_tree(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Trains the tree on X and returns the oracle y and the predicted y.
    """
    global CLASSIFIER

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=17)

    # Fitting
    CLASSIFIER = CLASSIFIER.fit(X_train, y_train)

    # Predicting
    y_pred = CLASSIFIER.predict(X_test)

    return y_test, y_pred


def save_decision_tree():
    """
    Saves a .sav file containing the model.
    """
    pickle.dump(CLASSIFIER, open(OUTPUT_DECISION_TREE, 'wb'))


def save_printed_decision_tree():
    """
    Saves a .pdf file with the printed decision tree.
    """
    global CLASSIFIER
    dot_data = StringIO()
    features = ('latitude', 'longitude', 'time_of_day', 'day_of_week', 'har_bike', 'har_bus', 'har_car', 'har_still', 'har_walk')
    targets_list = ("leisure", "restaurants", "sport")
    tree.export_graphviz(CLASSIFIER, out_file=dot_data, feature_names=features, class_names=targets_list)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    graph[0].write_pdf(OUTPUT_DECISION_TREE_PDF)  # must access graph's first element


def main():
    global CLASSIFIER
    har_dataset = pd.read_csv(INPUT_FILE_HAR)

    y_original = har_dataset.pop("place_category")
    X_original = har_dataset.drop(cols_to_drop, axis=1)

    X_oversampled, y_oversampled = oversample(X_original, y_original)

    # do dummy encoding
    har_cols = pd.get_dummies(X_oversampled["har"], prefix="har")
    one_hot_place_category = OneHotEncoder().fit_transform(y_oversampled.values.reshape(-1,1)).toarray()

    X_oversampled.drop(["har"], axis=1, inplace=True)
    X = pd.concat([X_oversampled, har_cols], axis=1)
    y = one_hot_place_category

    y_test, y_pred = train_tree(X, y)
    accuracy = accuracy_score(y_test, y_pred)
    number_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
    print(f"Decision tree depth:\t\t{CLASSIFIER.get_depth()}\nTotal number of samples:\t{len(X_oversampled)}\nCorrectly predicted samples:\t{number_correct_samples}\nAccuracy:\t\t\t{accuracy * 100}%")
    
    print("Saving the model for later use...")
    save_decision_tree()
    
    # print("Saving the decision tree (graphical)...")
    # save_printed_decision_tree()


if __name__ == '__main__':
    main()