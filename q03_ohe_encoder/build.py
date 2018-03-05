from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


def ohe_encode(X_train,X_test,category_index=category_index):
    X_cat= X_train.iloc[:,category_index]
    X_train=pd.concat([X_train,pd.get_dummies(X_cat,columns=X_cat.columns )], axis=1);
    X_train=X_train.drop(X_cat.columns, axis=1)

    X_cat_test= X_test.iloc[:,category_index]
    X_test=pd.concat([X_test,pd.get_dummies(X_cat_test,columns=X_cat_test.columns )], axis=1)
    X_test=X_test.drop(X_cat_test.columns, axis=1)
    return X_train, X_test
