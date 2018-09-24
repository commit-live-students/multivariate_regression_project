# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
def label_encode(x_train,x_test):
    labelencoder = LabelEncoder()


    X_transfrom = x_train.apply(labelencoder.fit_transform)
    X_test_transform = x_test.apply(labelencoder.fit_transform)
    return X_transfrom,X_test_transform
label_encode(x_train,x_test)


