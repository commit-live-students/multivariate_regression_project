# %load q04_data_visualisation/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
data = load_data('data/student-mat.csv') 
x_train, x_test, y_train, y_test =  split_dataset(data)
x_train,x_test = label_encode(x_train,x_test)

def visualise_data(data,figname = 'data/student-mat.csv'):
    train = pd.concat([x_train,y_train],axis=1)
    plt.scatter
    
    


c = visualise_data(data,figname = 'data/student-mat.csv')




