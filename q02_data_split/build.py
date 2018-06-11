#split_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

df = load_data('data/student-mat.csv')

def split_dataset(df1):
    x = df1.iloc[:,:-1]
    y = df1.iloc[:,-1]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    return x_train,x_test,y_train,y_test


