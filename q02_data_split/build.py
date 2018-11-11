# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
df = load_data('data/student-mat.csv')

def split_data(df1):
    X = df1.drop(df1.columns[len(df1.columns)-1], axis=1)
    y = df1.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3)
    return X_train, X_test, y_train, y_test





