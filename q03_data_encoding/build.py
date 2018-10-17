from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
def label_encode(X_train, X_test):
    num = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_variables = X_train.select_dtypes(include=num).columns.values
    all_columns = X_train.columns.values
    cat_variables = list(set(all_columns) - set(num_variables))
    le = LabelEncoder()
    for val in cat_variables:
        temp = pd.DataFrame(le.fit_transform(X_train[val]))
        X_train[val] = temp
        temp1 = pd.DataFrame(le.fit_transform(X_test[val]))
        X_test[val] = temp1

    return X_train, X_test




label_encode(x_train, x_test)
