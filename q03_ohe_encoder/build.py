from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train, x_test = label_encode(x_train, x_test)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


# Write your code below
def ohe_encode(X_train, X_test, category_index1=category_index):
    ohe = OneHotEncoder(categorical_features=category_index1, sparse=False)
    temp = pd.DataFrame(ohe.fit_transform(X_train))
    temp1 = pd.DataFrame(ohe.fit_transform(X_test))
    X_train.drop(X_train.columns[category_index1], axis=1, inplace=True)
    X_test.drop(X_test.columns[category_index1], axis=1, inplace=True)
    X_train = X_train.merge(temp, how='inner', left_index=True, right_index=True)
    X_test = X_test.merge(temp1, how='inner', left_index=True, right_index=True)
    return X_train, X_test
