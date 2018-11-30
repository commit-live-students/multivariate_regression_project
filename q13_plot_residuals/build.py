# %load q13_plot_residuals/build.py

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)

model = linear_regression(x_train, y_train)

y_pred, mse, mae, r2 = regression_predictor(model, x_train, y_train)

def plot_residuals(y_test, y_pred, name):
    
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals)
    plt.title('Residual Plot')
    plt.savefig('./images/data_image.png')
    
    
    plt.show();


