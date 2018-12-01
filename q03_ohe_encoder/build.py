# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']

# Write your code below
def ohe_encode(x_train, x_test, category_index=category_index):
    x_train,x_test = label_encode(x_train,x_test)
    oneHotEncd = OneHotEncoder(categorical_features=category_index, sparse=False)
    X_transform = oneHotEncd.fit_transform(x_train)
    X_test_transform = oneHotEncd.fit_transform(x_test)
    
    return X_transform, X_test_transform
    
X_transform, X_test_transform = ohe_encode(x_train, x_test)


