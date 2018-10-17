# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
path = 'data/student-mat.csv'
df = load_data(path)
x_train, x_test, y_train, y_test = split_dataset(df)
category_index1 = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']

def ohe_encode(x_train, x_test,category_index = category_index1):
    cat_col_list = x_train.iloc[:,category_index]

    for cat_col in cat_col_list:
        # encoding in dummy variable
        dummies = pd.get_dummies(x_train[cat_col], prefix=cat_col)
        x_train = pd.concat([x_train, dummies], axis=1)

        dummies1 = pd.get_dummies(x_test[cat_col], prefix=cat_col)
        x_test = pd.concat([x_test, dummies1], axis=1)

        # removing the  variable
        x_train.drop(cat_col, axis=1, inplace=True)
        x_test.drop(cat_col, axis=1, inplace=True)
    return x_train,x_test

