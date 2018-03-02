from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

def label_encode(X_train,X_test):
    le = LabelEncoder()
    numeric = X_train._get_numeric_data().columns.tolist()
    all_columns = x_train.columns.tolist()
    non_numeric = [item for item in all_columns if item not in numeric]

    for column in non_numeric:
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.fit_transform(X_test[column])
    return X_train, X_test    
