# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

def label_encode(X_train,X_test):

    #Encoding x_train
    columnsToEncode = list(x_train.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        x_train[feature] = le.fit_transform(x_train[feature])
    x_train_transform = x_train

    #Encoding x_test
    columnsToEncode = list(x_test.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        x_test[feature] = le.fit_transform(x_test[feature])
    x_test_transform = x_test

    return x_train_transform,x_test_transform

label_encode(x_train,x_test)
