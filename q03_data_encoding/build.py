from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
def label_encode(xtrain,xtest):
    columnsToEncode_xtrain = list(xtrain.select_dtypes(include=['category','object']))
    columnsToEncode_xtest = list(xtest.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode_xtrain:
        try:
            xtrain[feature] = le.fit_transform(xtrain[feature])
        except:
            pass
        for feature in columnsToEncode_xtest:
            try:
                xtest[feature] = le.fit_transform(xtest[feature])
            except:
                pass
    return xtrain,xtest
