# %load q01_load_data/build.py
import pandas as pd

# Write your code below
# path = 'data/student-mat.csv'

def load_data(path):
    return pd.read_csv(path, sep = ';')
# load_data(path)


