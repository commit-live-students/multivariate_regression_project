from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
df = load_data('data/student-mat.csv')
def split_dataset(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return train_test_split(X,y, random_state=9, test_size =0.2)

print (split_dataset(df))
