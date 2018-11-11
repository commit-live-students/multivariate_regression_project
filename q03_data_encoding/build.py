# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

def label_encode(X_train, X_test):
    numeric_features = [a for a in range(len(df.dtypes)) if df.dtypes[a] in ['int64','float64']]
    cat_features = df.columns.difference(df.columns[numeric_features])
    label_encoder = LabelEncoder()
    X_transform = X_train.copy()
    X_test_transform = X_test.copy()
    for feature in cat_features:
        X_transform[feature] = label_encoder.fit_transform(X_train[feature])
        X_test_transform[feature] = label_encoder.fit_transform(X_test[feature])
    return X_transform, X_test_transform



