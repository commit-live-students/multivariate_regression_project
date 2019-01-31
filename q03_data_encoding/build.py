# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
def label_encode(X_train, X_test):
    label = LabelEncoder()
    for col in x_train.columns:
        X_train[col] = label.fit_transform(X_train[col])
        X_test[col] = label.fit_transform(X_test[col])
    
    X_train_transform = X_train
    X_test_transform = X_test
    return X_train_transform, X_test_transform
    





