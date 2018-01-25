#Write a function "linear_model" to create an end-to-end model of regression including creation,cross validation and prediction

Write a function create_model which takes dataset splits as input and outputs the model and testing predictions.
Feel free to output other forms of error measures like calculated mse,mae,r2 scores in any form you like.
This will help you in comparing various models and approaches.

Writing such function promotes to the reusability of code and it is very convenient.

#### Parameters:

| x_train | DataFrame | compulsory |  | Independent variables training |
| x_test| DataFrame | compulsory |  | Independent variables testing|
| y_train | DataFrame | compulsory |  | training target|
| y_test| DataFrame | compulsory |  | testing target|
| name|  | compulsory |  | name for stats|



#### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| G| | Model |
| y_pred | | predictions|
|stats|dataframe|store results-'cross_validation', 'rmse','mae','r2' with index as name
