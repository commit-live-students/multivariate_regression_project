import pandas as pd

# Write your code below

def load_data(path):
    df =  pd.read_csv(path,sep=';')
    return df
