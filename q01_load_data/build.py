import pandas as pd

# Write your code below

def load_data(a):
    df =  pd.read_table(a,sep=';')
    return df
