# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
def label_encode(X,X_test):
    X_transform = X.copy()
    X_test_transform = X_test.copy()
    labelencoder = LabelEncoder()
    categorical_variables = list(X.select_dtypes(exclude=['number']))
    for variable in categorical_variables:
        X_transform[variable] = labelencoder.fit_transform(X_transform[variable])
        X_test_transform[variable] = labelencoder.fit_transform(X_test_transform[variable])
    return X_transform,X_test_transform
label_encode(x_train,x_test)


