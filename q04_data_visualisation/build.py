# %load q04_data_visualisation/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
# %matploylib.inline

data = load_data('data/student-mat.csv') 
X_train, X_test, y_train, y_test =  split_dataset(data)
X_train,X_test = label_encode(X_train,X_test)

# Write your code below
def visualise_data(data,figname):
    
    scatter_matrix(data, alpha=0.2, figsize=(15,15), diagonal='kde')
    plt.show()
    
# visualise_data(data,'figname')


