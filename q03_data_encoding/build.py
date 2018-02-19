from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

def label_encode(X_train,X_test):
    """encodes the non-numeric values to numeric"""
    cat_columns = X_train.select_dtypes(exclude=[np.number]).columns
    X_train.loc[:,cat_columns]=X_train[cat_columns].apply( LabelEncoder().fit_transform)
    X_test.loc[:,cat_columns]=X_test[cat_columns].apply( LabelEncoder().fit_transform)
    return X_train, X_test
