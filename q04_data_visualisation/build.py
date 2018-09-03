# %load q04_data_visualisation/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
data = load_data('data/student-mat.csv') 
x_train, x_test, y_train, y_test =  split_dataset(data)
x_train,x_test = label_encode(x_train,x_test)

# Write your code below
def visualise_data(df):
    data_1 = df
    train_1 = pd.concat([x_train, y_train], axis = 1)
    (scatter_matrix(train_1, alpha=0.2, diagonal='hist'))
    plt.show()
    #return(axs)

visualise_data(data)


