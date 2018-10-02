# %load q01_load_data/build.py
import pandas as pd

def load_data(path = 'data/student-mat.csv'):
    df = pd.read_table(path,sep =';')
    return df
load_data()



