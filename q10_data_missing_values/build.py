# %load q10_data_missing_values/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
import numpy as np
np.random.seed(9)
import pandas as pd
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)

# Write your code below
def describe_df(x_train):
    #print(x_train.shape)
    descD=x_train.describe()
    #x_train.info()
    #abc=x_train.columns.apply(pd.value_counts(x_train[x]))
    #print(x_train['absences'].value_counts())
    vc=x_train.apply(pd.value_counts)
    return descD,vc
#describe_df(x_train)

