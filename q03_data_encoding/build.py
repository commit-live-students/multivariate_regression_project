# %load q03_data_encoding/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = load_data('data/student-mat.csv')
pd.set_option('display.max_columns', 100)
 
x_train, x_test, y_train, y_test =  split_dataset(df)

# Write your code below
def label_encode(train, test):
    #data_1 = data
    le = LabelEncoder()
    for col in test.columns.values:
       # Encoding only categorical variables
       if test[col].dtypes=='object':
            # Using whole data to form an exhaustive list of levels
            data=train[col].append(test[col])
            le.fit(data.values)
            train[col]=le.transform(train[col])
            test[col]=le.transform(test[col])
    return (train, test)




label_encode(x_train, x_test)
df['Pstatus'].value_counts()


