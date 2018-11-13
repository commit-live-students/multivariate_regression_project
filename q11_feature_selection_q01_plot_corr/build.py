# %load q11_feature_selection_q01_plot_corr/build.py

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)

def plot_corr(df, size=11):
    x_train, x_test, y_train, y_test =  split_dataset(df)
    df_train = pd.concat([x_train,y_train], axis=1)
    corr = df_train.corr()
    fig, ax = subplots(figsize=(size,size))
    plt.set_cmap('YlOrRd')
    ax.matshow(corr)
    xticks(range(len(corr.columns)), corr.columns, rotation=90)
    yticks(range(len(corr.columns)), corr.columns)
    fig.savefig('./images/data_image.png')
    return ax



