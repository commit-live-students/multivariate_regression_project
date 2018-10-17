
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data


from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset



from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


# =============================================================================
# To visualise data, you need to pass training data only as the assumption holds that test set is unknown data and obviously,you cant not make decision based on unseen data :-p

#Remember to concatenate training features and labels if you want to check that scatterplots which I would prefer.You are free to explore labels to labels, features to features ,etc scatterplots as you want by passing arguments
#============================================================================
#visualise_data(pd.concat([x_train,y_train],axis=1),"../images/data_image.png")

# Write your solution here:
def plot_corr(df,size=11):
    numerics = ['int8','int16','int32','int64','float16','float32','float64']
    num_cols = list(df.select_dtypes(include=numerics).columns)
    corr1 = pd.DataFrame(index=num_cols,columns=num_cols)
    for i in range(0,len(num_cols)):
        for j in range(0,len(num_cols)):
            corr1.loc[num_cols[i],num_cols[j]] = df[num_cols[i]].corr(df[num_cols[j]])

    plt.pcolor(corr1)
    plt.yticks(np.arange(0.5, len(corr1.index), 1), corr1.index)
    plt.xticks(np.arange(0.5, len(corr1.columns), 1), corr1.columns)
    plt.show()


#plot_corr(df)
