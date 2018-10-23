# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']

# one = OneHotEncoder(categorical_features=category_index)
# X_transform = one.fit_transform(x_train)
# X_transform_test = one.fit_transform(x_test)
def ohe_encode(X,X_test,category_index=[0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]):
    X_transform = X.copy()
    X_test_transform = X_test.copy()
    ohe = OneHotEncoder(categorical_features=category_index)
    X_transform = ohe.fit_transform(X_transform)
    X_test_transform = ohe.fit_transform(X_test_transform)
    return X_transform,X_test_transform







