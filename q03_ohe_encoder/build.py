# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
np.random.seed(7)
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


def ohe_encode(x_train,x_test,category_index=category_index):
    columnsToEncode = list(x_train.select_dtypes(include=['category','object']))
    for feature in columnsToEncode:
        x_train[feature] = pd.get_dummies(x_train[feature])
    x_train_transform = x_train

    #Encoding x_test
    columnsToEncode = list(x_test.select_dtypes(include=['category','object']))
    for feature in columnsToEncode:
        x_test[feature] = pd.get_dummies(x_test[feature])
    x_test_transform = x_test
    return x_train,x_test

ohe_encode(x_train,x_test,category_index=category_index)
