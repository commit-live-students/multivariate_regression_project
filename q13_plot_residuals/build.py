import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_residuals(y_test,y_pred,name):
    error_residuals = y_test-y_pred
    plt.scatter(y_test,error_residuals)
    plt.title(name)
    plt.xlabel("Actual Values")
    plt.ylabel("Residual Error of predicted values")
    plt.show()
