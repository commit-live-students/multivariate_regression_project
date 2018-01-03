# Separate Data into Feature and Target Variables - Setting up the analysis

Now that you have imported the data, let's split data into target or dependent variables and features or independent variables. We can use these variables later on to fit a linear regression model.

What would be the dependent variables here?
-G3

Tip: In practice, we denote dependent variables with capital X and target variables with small y.
Now split this to training and testing set
## Write a function `split_data` that:
-Takes dataframe as input and splits into x_train,x_test,y_train,y_test

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| df | DataFrame | compulsory |  | Input Dataframe to split |


### Returns:
| Return | dtype | description |
| --- | --- | --- | 
| x_train | DataFrame | compulsory |  | Independent variables training |
| x_test| DataFrame | compulsory |  | Independent variables testing|
| y_train | DataFrame | compulsory |  | training target|
| y_test| DataFrame | compulsory |  | testing target|
