# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
df = load_data('data/student-mat.csv')
df1 = df.copy()
# Write your code below
def split_dataset(df):
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    x_test,x_train,y_test,y_train = train_test_split(X,y,test_size = 0.8,random_state = 42)
    return x_train,x_test,y_train,y_test



