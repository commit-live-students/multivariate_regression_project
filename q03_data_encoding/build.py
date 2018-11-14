# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

le = preprocessing.LabelEncoder()

def label_encode(x_train,x_test):
    X_transform = x_train.select_dtypes(include=['object']).copy()
    X_test_transform = x_test.select_dtypes(include=['object']).copy()
    for i in X_transform:
        x_train[i]=le.fit_transform(X_transform[i])
    
    for j in X_test_transform:
        x_test[i]=le.fit_transform(X_test_transform[i])
        
    return x_train,x_test
        
        
        
    
    
    
    



label_encode(x_train,x_test)


