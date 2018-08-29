# %load q01_load_data/build.py
import pandas as pd
from sklearn.utils import shuffle


# Write your code below
def load_data(path):
    df=pd.read_table(path,sep=';')
    df = shuffle(df)
    return df

#path='data/student-mat.csv'    
#load_data(path)
    


