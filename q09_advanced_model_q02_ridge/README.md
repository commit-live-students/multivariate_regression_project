#Tryout Ridge with multioutput

So now just replace your multioutput regression function with Ridge and chek out the results.
This technique of trying various models and compare against baselinbe is helpful in determing the optimum solution.

#Write function Ridge reusing create_model function and passing appropriate paramters as instructed above
your function signature will look like this and you will perform following operations
1. Split X,y in x_train,x_test,y_train,y_test
2. Create model by calling create_model(Note: In Ridge model, make sure you make the 'Normalize' parameter as True)
3. Get its predictions on entire set and record them in res creating a model predicted feature set
4. store all statistical error metrics in stats dataframe
#### Parameters:
|---|---|---|---|---|
| x_train | DataFrame | compulsory |  | Independent variables training |
| x_test| DataFrame | compulsory |  | Independent variables testing|
| y_train | DataFrame | compulsory |  | training target|
| y_test| DataFrame | compulsory |  | testing target|
| name|  | compulsory |  | name for stats|
| alpha|  |  |  | |
#### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| G| | Model |
| y_pred | | predictions|
|stats|dataframe|store results-'cross_validation', 'rmse','mae','r2' with index as name
