# %load q05_linear_regression_model/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train, x_test = label_encode(x_train,x_test)

y = df.loc[:, df.columns == 'G3']
X = df.loc[:, df.columns != 'G3']

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
X = pd.DataFrame(X).fillna(0)

# Write your code below
def linear_regression(X,y):
    lm = LinearRegression()
    #lm.fit(X[:, np.newaxis], y)
    lm.fit(pd.DataFrame(X), pd.DataFrame(y))
    return lm
    
    
    



