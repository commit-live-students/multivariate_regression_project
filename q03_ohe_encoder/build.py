# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

from unittest import TestCase
from inspect import getargspec

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']
print category_index

def ohe_encode(X,X_test,category_index=category_index):
    X_cat_train = X.iloc[:,category_index]
    x = X.drop(X_cat_train.columns, axis=1)
    x = pd.concat([x,pd.get_dummies(X_cat_train,drop_first=True)],axis=1)
    
    X_cat_test = X_test.iloc[:,category_index]
    x_test = X_test.drop(X_cat_test.columns, axis=1)
    x_test = pd.concat([x_test,pd.get_dummies(X_cat_test,drop_first=True)],axis=1)
    
    return x_train, x_test

args = getargspec(ohe_encode)
print len(args[0])
print args[3]


    

    

    
   


