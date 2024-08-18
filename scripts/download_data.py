import pickle

import datasets
import yaml
import pandas as pd
import time
import datetime

def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_json("Luxury_Beauty_5.json", lines=True)
    df = df.head(1000)
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump(obj=df, file=f)


main()