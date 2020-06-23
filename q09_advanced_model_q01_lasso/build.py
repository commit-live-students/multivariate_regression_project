# %load q09_advanced_model_q01_lasso/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

def lasso(x_train, x_test, y_train, y_test, alpha=0.1):
    
    lasso_model = Lasso(alpha)
    G = lasso_model.fit(x_train, y_train)
    val = cross_validation_regressor(lasso_model,x_train,y_train)
    y_pred, mse, mae, r2 = regression_predictor(lasso_model, x_test, y_test)
    r2 = r2_score(y_test, y_pred)
    stat_table = pd.DataFrame([[val, mae, r2, mse]], columns=['cross_validation', 'mae', 'r2', 'rmse'])
    
    return G, y_pred, stat_table


    



