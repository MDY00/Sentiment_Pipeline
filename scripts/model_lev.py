import pickle

import datasets
import yaml
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import dvc.api
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
import json
import joblib
import mlflow
def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    grid_search = joblib.load('data/grid_search_model.joblib')
    randomized_search = joblib.load('data/randomized_search_model.joblib')
    halvingrandom_search = joblib.load('data/halvingrandom_search_model.joblib')
    halvinggrid_search = joblib.load('data/halvinggrid_search_model.joblib')
    pipeline = joblib.load('data/pipe.joblib')
    with open("data/X_train.pkl", "rb") as f:
        X_train = pickle.load(file=f)
    with open("data/X_test.pkl", "rb") as f:
        X_test = pickle.load(file=f)
    with open("data/y_train.pkl", "rb") as f:
        y_train = pickle.load(file=f)
    with open("data/y_test.pkl", "rb") as f:
        y_test = pickle.load(file=f)

    print("Gridsearch")
    print(f1_score(y_test, grid_search.predict(X_test), average='macro'))
    print("Randomized")
    print(f1_score(y_test, randomized_search.predict(X_test), average='macro'))
    print("halvingrandom")
    print(f1_score(y_test, halvingrandom_search.predict(X_test), average='macro'))
    print("halvinggrid")
    print(f1_score(y_test, halvinggrid_search.predict(X_test), average='macro'))


    metricsx = {"tmp": "tmp"}

    with open('data/score.json', 'w') as f:
        json.dump(metricsx, f)

main()