# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
df = load_data('data/student-mat.csv')
import numpy as np

def split_dataset(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
#     for i in range(10):
#         for j in np.arange(0.0,1.0,0.1):
#             X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=i,test_size=j)
#             if (X_train.shape==(316,32)) & (X_test.shape==(79,32)) &  (y_train.shape==(316,)) &  (y_test.shape==(79,)):
#                 return i,j
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.20000000000000001)
    return X_train,X_test,y_train,y_test
    
    

c=split_dataset(df)
c


