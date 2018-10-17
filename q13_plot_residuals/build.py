# %load q13_plot_residuals/build.py

from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)
import matplotlib.pyplot as plt
l2=Ridge(alpha=0.01)
l2.fit(x_train,y_train)
# Write your code below
def plot_residuals(model, x_test, y_test):
    y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)
    error_residuals=y_test-y_pred
    plt.scatter(y_test,error_residuals)
    plt.title('Residual Plot')
    plt.xlabel('SalePrice')
    plt.ylabel('Errors')
    plt.show()

#plot_residuals(l2, x_test, y_test)

