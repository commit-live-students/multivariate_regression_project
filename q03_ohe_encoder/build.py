from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']


# Write your code below
def ohe_encode(xtr,xts):
    for i in xtr.columns:
        if i in category_index:
            xtr = pd.get_dummies(xtr, drop_first=True)
    for i in xts.columns:
        if i in category_index:
            xts = pd.get_dummies(xts, drop_first=True)
    return xtr,xts
print (ohe_encode(x_train,x_test))
