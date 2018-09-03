# %load q02_data_split/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
df = 'data/student-mat.csv'
data_1 = load_data(df)

# Write your code below
def split_dataset(data):
    
    #data_1 = pd.read_csv(data,sep=';')
    X = data_1.iloc[:, :-1]
    y = data_1.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    return(x_train, x_test, y_train, y_test)
    
split_dataset(df)


