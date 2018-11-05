# %load q01_load_data/build.py
import pandas as pd

path='data/student-mat.csv'

def load_data(path):
    df=pd.read_csv(path,sep=';')
    return df
    

load_data('data/student-mat.csv')



