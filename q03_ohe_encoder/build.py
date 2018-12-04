# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd

path = 'data/student-mat.csv'
df = load_data(path)
category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']
columns = [col for col in (df.columns) if df[col].dtype == 'object']
#print(df.shape)
print(category_index)
df_new = pd.get_dummies(df, columns=columns)
X_train, X_test, y_train, y_test = split_dataset(df_new)

def ohe_encode(X_train, X_test, defaults=category_index):
    X_transform, X_test_transform = label_encode(X_train, X_test)
    return X_transform, X_test_transform

ohe_encode(X_train, X_test, category_index)


