# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)


def label_encode(x_train, x_test):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    X_transform = x_train.copy()
    X_test_transform = x_test.copy()
    for feature in columnsToEncode:
        X_transform[feature] = le.fit_transform(X_transform[feature])
        X_test_transform[feature] = le.fit_transform(X_test_transform[feature])
    
    return X_transform, X_test_transform
    
label_encode(x_train, x_test)
    
    





