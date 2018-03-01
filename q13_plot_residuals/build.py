# %load q13_plot_residuals/build.py


import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
#from greyatomlib.linear_regression.q05_residuals.build import residuals
#from greyatomlib.multivariate_regression_project.q06_cross_validation import cross_validation_regressor

from sklearn.linear_model import LinearRegression
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train, x_test = label_encode(x_train,x_test)
lin_reg = linear_regression(x_train,y_train)
y_pred,_,__,___ = regression_predictor(lin_reg,x_test,y_test)


def plot_residuals(y_test,y_pred,name):
    error_residuals = y_test - y_pred
    stats.probplot(error_residuals, dist="norm", plot=pylab)
    return pylab.show()

#plot_residuals(y_test,y_pred,'name')
