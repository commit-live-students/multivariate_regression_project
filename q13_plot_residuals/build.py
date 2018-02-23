

import matplotlib.pyplot as plt
#%matplotlib inline
def plot_residuals(y_test,y_pred,name):
    residuals = y_test - y_pred
    plt.scatter(y_test,residuals)
    plt.title(name)
