

import matplotlib.pyplot as plt

def plot_residuals(y_test,y_pred,name):
    plt.scatter(y_test, y_test-y_pred)
    plt.title(name)
    plt.xlabel('Actual Y')
    plt.ylabel('Residual Error')
    plt.show()
