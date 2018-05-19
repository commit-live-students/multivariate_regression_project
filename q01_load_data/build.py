# %load q01_load_data/build.py
import pandas as pd

# Write your code below
def load_data(path):
    return pd.read_csv(path,sep=';')


