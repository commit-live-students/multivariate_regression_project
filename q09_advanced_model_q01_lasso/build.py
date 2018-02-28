# %load q09_advanced_model_q01_lasso/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

# Write your solution here
def lasso(x_train, x_test, y_train, y_test,alpha=0.1):
    model =Lasso(alpha=alpha)
    model.fit(x_train,y_train)
    val = cross_validation_regressor(model,x_train,y_train)
    y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)
    stats = pd.DataFrame()
    stats['CV_score'] = val, val
    stats['MAE'] = mae
    stats['r2'] = r2
    stats['MSE'] = np.sqrt(mse)
    #stats.set_index('Name',inplace=True)
    return model, y_pred, stats

lasso(x_train, x_test, y_train, y_test,alpha=0.1)
