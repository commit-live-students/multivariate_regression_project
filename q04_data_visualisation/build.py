from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
plt.switch_backend('agg')
data = load_data('data/student-mat.csv') 

x_train, x_test, y_train, y_test =  split_dataset(data)
x_train,x_test = label_encode(x_train,x_test)
train_1 = pd.concat([x_train, y_train], axis = 1)

# Write your code below
def visualise_data(data,figname):
  
    
    scatter_matrix(data, alpha=0.2, diagonal='kde')
    plt.show()



