# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
    
def label_encode(x_train, x_test):

    le = LabelEncoder()
    X_col = list(df.select_dtypes(include=['category','object']))
    Xtest_col = list(df.select_dtypes(include=['category','object']))

    for i in X_col:
        x_train[i] = le.fit_transform(x_train[i])
    

    for j in Xtest_col:
        x_test[j] = le.fit_transform(x_test[j])
        
    return x_train, x_test  



# def label_encode(X, X_test):

le = LabelEncoder()
X_col = list(df.select_dtypes(include=['category','object']))
Xtest_col = list(df.select_dtypes(include=['category','object']))

for i in X_col:
    x_train[i] = le.fit_transform(x_train[i])
    

for j in Xtest_col:
    x_test[j] = le.fit_transform(x_test[j])
        
#     return X, X_test
type(label_encode(x_train, x_test))
x_train.shape
x_test.shape


