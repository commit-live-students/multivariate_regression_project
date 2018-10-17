# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split as tts
import pandas as pd
df = load_data('data/student-mat.csv')

def split_dataset(df):
    X = df.drop('G3',axis = 1)
    y = df['G3']
    x_train,x_test,y_train,y_test = tts(X,y,test_size = 0.2)
    return x_train, x_test, y_train, y_test

    
split_dataset(df)    


df


