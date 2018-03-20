from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

def label_encode(X, X_test):

    le = LabelEncoder()
    columnsToEncode_X = list(df.select_dtypes(include=['category','object']))
    columnsToEncode_Xtest = list(df.select_dtypes(include=['category','object']))

    for feature in columnsToEncode_X:
            try:
                X[feature] = le.fit_transform(X[feature])
            except:
                print('Error encoding '+feature)

    for feature in columnsToEncode_Xtest:
            try:
                X_test[feature] = le.fit_transform(X_test[feature])
            except:
                print('Error encoding '+feature)


    return X, X_test
