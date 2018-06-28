# %load q01_load_data/build.py
import pandas as pd
import numpy as np
# Write your code below
def load_data(path):
    np.random.seed(7)
    df = pd.read_table(path, sep=';')
    return df



