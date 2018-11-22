# %load q09_advanced_model_q02_ridge/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

# Write your code below
def ridge(x_train, x_test, y_train, y_test,alpha=0.1):
    model = Ridge(alpha=alpha)
    model.fit(x_train,y_train)
    val = cross_validation_regressor(model,x_train,y_train)
    y_pred, mse, mae, r2 = regression_predictor(model,x_test,y_test)
    stats = pd.DataFrame(columns=['cross_validation', 'mae','r2','rmse'])
    stats.loc[0,'cross_validation'] = val
    stats.loc[0,'rmse'] = mse **(0.5)
    stats.loc[0,'mae'] =  mae
    stats.loc[0,'r2'] = r2
    return model, y_pred, stats

    




