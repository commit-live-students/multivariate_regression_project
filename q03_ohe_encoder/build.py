# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']

def ohe_encode(x_train, x_test, ct = category_index):
    
    X_transform = pd.get_dummies(x_train.iloc[ct], drop_first=True)
    X_test_transform = pd.get_dummies(x_test.iloc[ct], drop_first=True)
    
    return X_transform, X_test_transform

ohe_encode(x_train, x_test, category_index)

    
   






