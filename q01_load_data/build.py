# %load q01_load_data/build.py
import pandas as pd

# Write your code below
path = 'data/student-mat.csv'
def load_data(path):
    df = pd.read_table(path, sep=';')
    return df
load_data(path)

