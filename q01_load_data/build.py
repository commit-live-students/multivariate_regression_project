# %load q01_load_data/build.py
import pandas as pd

def load_data(path):
    df = pd.read_csv(filepath_or_buffer=path, delimiter=';')
    return df



