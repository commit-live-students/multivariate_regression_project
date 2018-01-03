# Default imports
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
import numpy as np
import matplotlib.pyplot as plt
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
import numpy as np
import pandas as pd
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)


np.random.seed(9)
def percentile_k_features(X,y, k=50):
    model = f_regression
    skb = SelectPercentile(model, k)
    predictors = skb.fit_transform(X,y)
    scores = list(skb.scores_)
    
    top_k_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:predictors.shape[1]]
    top_k_predictors = [X.columns[i] for i in top_k_index]
    
    X_indices = np.arange(X.shape[1])
    scores = -np.log10(skb.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)',  color='darkorange', edgecolor='black')
    plt.xticks(X_indices+np.ones(X_indices.shape), X.columns,rotation=90 )
    plt.savefig("images/select_percentile_graph.png")
    return top_k_predictors




