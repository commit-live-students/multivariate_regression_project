# %load q01_load_data/build.py
import pandas as pd

# Write your code below
def load_data(path):
    df = pd.read_csv(path,sep=';')
    df = df.sample(n=df.shape[0],random_state=7)
    return df


