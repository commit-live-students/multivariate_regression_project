import pandas as pd

#path = '../data/student-mat.csv'
    
def load_data(path):
    return pd.read_csv(path, sep = ';')
