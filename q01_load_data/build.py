# %load q01_load_data/build.py
import pandas as pd

path = 'data/student-mat.csv'
# Write your code below
def load_data(data):
    data_1 = pd.read_csv(data, sep = ';')
    return (data_1)

(load_data(path))


