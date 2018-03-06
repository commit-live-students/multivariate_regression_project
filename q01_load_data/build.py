import pandas as pd

path = './data/student-mat.csv'
def load_data(path):
    df = pd.read_csv(path,sep=';')
    return df
