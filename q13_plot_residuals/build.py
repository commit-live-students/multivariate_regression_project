import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Write your code below
def plot_residuals(y_actual,y_pred,name):
    plt.plot(y_actual,y_pred)
    plt.show()

#plot_residuals([1,2,3,4],[4,5,6,7],'hello')
