from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


def ohe_encode(X,X_test,category_index=category_index):
    ohe = OneHotEncoder(categorical_features=category_index, sparse=False)
    X_transform = ohe.fit_transform(X)
    X_test_transform = ohe.transform(X_test)
    return X_transform, X_test_transform


    

    
   
