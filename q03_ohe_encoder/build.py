# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']



def ohe_encode(df,x_test,category_index):
    ind=[]
    for i in range(len(list(df))):
        ind.append(i)

    a=[]
    for i,j in zip(list(df),ind):
        a.append((j,i))

    main=[]    
    for i in a:
        for j in category_index:
            if i[0]==j:
                main.append(i[1])

    X_train = pd.get_dummies(x_train)
    X_test = pd.get_dummies(x_test)
   
    return X_train,X_test


c=ohe_encode(x_train,x_test,category_index)
c


