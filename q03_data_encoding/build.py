# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)
X_transform = x_train
X_test_transform  = x_test
# Write your code below
def label_encode(x_train, x_test):
    lb_make = LabelEncoder()
    a = x_train.select_dtypes(exclude=[np.number]).columns.values
    for i in range(len(a)):
        X_transform[a[i]]= lb_make.fit_transform(x_train[a[i]])
    b = x_train.select_dtypes(exclude=[np.number]).columns.values
    for i in range(len(b)):
        X_test_transform[b[i]]= lb_make.fit_transform(x_test[b[i]])
    return X_transform,X_test_transform
  


