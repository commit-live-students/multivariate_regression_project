# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
df = load_data('data/student-mat.csv')
def split_dataset(df):
    X= df.iloc[:,:-1]
    y=df['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=0.80)
    return X_train, X_test, y_train, y_test
split_dataset(df)
