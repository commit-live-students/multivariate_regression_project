import pandas as pd

path = '/home/darshind/Workspace/code/multivariate_regression_project/data/student-mat.csv'
def load_data(path):
    df = pd.read_csv(path,sep =';')
    return df
print load_data(path)
