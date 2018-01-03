# Predictor your model.

In this assignment you will write a regression predictor

 
##  Write a function `regression_predictor` that :
- Calculates the y_pred, mse, mae, r2

#### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| Model | | compulsory |  | Model to be fitted |
| X | Dataframe for training, testing; any format acceptable by sklearn| compulsory |  | X_test |
| y | Dataframe for training, testing; any format acceptable by sklearn | compulsory |  | y_train |



#### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| y_pred| | Predictions |
| mse | | mean squared error|
| mae | | mae|
| r2 | | r squared error|coefficient of determination|