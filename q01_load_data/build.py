import pandas as pd
import random

random.seed(7)

def load_data(path):
    df = pd.read_table(path, sep=';')
    
    return df
    




