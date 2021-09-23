import pickle
import pandas as pd

# Saved Model File
SAVED_MODEL_FILE = "09_decision_tree/decisiontree.sav"


def main():
    loaded_model = pickle.load(open(SAVED_MODEL_FILE, 'rb'))


if __name__ == '__main__':
    main()
