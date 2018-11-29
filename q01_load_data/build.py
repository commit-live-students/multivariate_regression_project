# %load q01_load_data/build.py
import pandas as pd

path = 'data/student-mat.csv'

# Write your code below
def load_data(path):
    return pd.read_table(path, sep=';')
    
load_data(path)


