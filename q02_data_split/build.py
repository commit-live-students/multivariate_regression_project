# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
df = load_data('data/student-mat.csv')

# Write your code below
def split_dataset(df):
    x_train,x_test,y_train,y_test = train_test_split(df.loc[:, 'school':'G2'],df.loc[:,'G3'],test_size=0.2)
    return x_train,x_test,y_train,y_test









