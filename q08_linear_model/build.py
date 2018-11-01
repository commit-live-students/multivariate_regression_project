# %load q08_linear_model/build.py
import pandas as pd
import numpy as np
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor


df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
model =linear_regression(x_train,y_train)
val = cross_validation_regressor(model,x_train,y_train)
y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)

# Write your code below
def linear_model(x_train, x_test, y_train, y_test):
    G = linear_regression(x_train, y_train)
    stats = pd.DataFrame([(val,mae,mse,r2)], columns = ['cross_val','rmse','mae','r2'])
    return G, y_pred, stats

linear_model(x_train,x_test,y_train,y_test)
    



