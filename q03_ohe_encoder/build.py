# %load q03_ohe_encoder/build.py


from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

import numpy as np

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


def ohe_encode(X,X_test,category_index=category_index):
    X1,X1_test=label_encode(X,X_test)
    ohe = OneHotEncoder(categorical_features=category_index,sparse=False)
    ohe.fit(X)
    X_transform=ohe.transform(X1)
    X_test_transform=ohe.transform(X1_test)            
    
    return pd.DataFrame(X_transform),pd.DataFrame(X_test_transform)
    
    
    
#ohe_encode(x_train, x_test,category_index)
    

