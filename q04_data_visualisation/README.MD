

# Encoding of Nominal and String attribues

Check for the attributes that indicate categorical values. Also you will find atrributes that are expressed as string or characters.

## Write a function `visualise_data()`:
use from pandas.plotting import scatter_matrix
make sure you have performed splitting before this exercise.
Create a relationship matrix between each of the 2 components.
I would recommend joining x_train and y_train before sending to this function to capture realtionship of every features with target variables.
To interpret these reults, go through following link
http://support.minitab.com/en-us/minitab-express/1/help-and-how-to/graphs/scatterplot/interpret-the-results/key-results/

features being linearly related could introduce colinearity means a problem.

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| data| DataFrame | compulsory |  |dataframe|
| figname | String | compulsory |  | path |
### Returns:
| Return | dtype | description |
| --- | --- | --- |
| plt|  |  |
