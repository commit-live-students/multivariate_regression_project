# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


# Write your code below
def ohe_encode(X,X_test,category_index=([0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22])):
    ohe_enc = OneHotEncoder()
    label_enc = LabelEncoder()
    for i in range(len(category_index)):
        X.iloc[:,category_index[i]] = label_enc.fit_transform(X.iloc[:,category_index[i]])
        X_test.iloc[:,category_index[i]] = label_enc.fit_transform(X_test.iloc[:,category_index[i]])

    return ohe_enc.fit_transform(X),ohe_enc.fit_transform(X_test)



