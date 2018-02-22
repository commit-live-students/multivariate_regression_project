import pandas as pd

path = 'data/student-mat.csv'
def load_data(path):
    data = pd.read_table(path,sep=';')
    return data
