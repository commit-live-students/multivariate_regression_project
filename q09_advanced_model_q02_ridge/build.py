# %load q09_advanced_model_q02_ridge/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

def ridge_model(x_train, x_test, y_train, alpha=0.1):
    model = Ridge(alpha)
    G = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    val = cross_validation_regressor(model,x_train,y_train)
    y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)
    stats1 = pd.DataFrame([[val, mae, mse,  r2]], columns=['cross_validation', 'mae', 'mse', 'r2'])
    return G, y_pred, stats1



