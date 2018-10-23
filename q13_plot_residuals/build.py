# %load q13_plot_residuals/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
import matplotlib
matplotlib.use('agg')
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

rid = Ridge(random_state=9,alpha=0.1)
model = rid.fit(x_train,y_train)
y_pred = model.predict(x_test)
name=df

import matplotlib.pyplot as plt
def plot_residuals(y_test,y_pred,name):
    
    plt.scatter(y_test,y_pred)
    plt.show()
c= plot_residuals(y_test,y_pred,name)
c


