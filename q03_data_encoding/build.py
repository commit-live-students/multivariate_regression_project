from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

def label_encode(X_train,X_test):
    """encodes the non-numeric values to numeric"""
    le = LabelEncoder()
    X_transform = pd.DataFrame()
    X_test_transform = pd.DataFrame()
    for col in X_train.columns.values:
        #print X_test[col]
        X_transform[col] = le.fit_transform(X_train[col])
        X_test_transform[col] = le.fit_transform(X_test[col])

    return X_transform,X_test_transform
