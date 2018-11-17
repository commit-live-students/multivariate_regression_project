# Perform feature Selection

Please note that both sub-exercises in this assignment are already completed by you in your previous explorations.
I have included them to show you how to analyse the output for your benefits.


## Write a function `feature_selection` that:
- Select best features with implementation of k percentile method. A function call to q02_best_k_features will do.
- return the features corresponding to the given best selected features


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| X |  | compulsory |  | Independent variable |
| y |  | compulsory |  | Dependent variable   |
| k| integer | compulsory | 50 | number as int for number of best features falling under k-th percentile |


### Feel free to play with k to obtain optimum results

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
|reduced features |list| k best feature list|

You will observe that most of features dropped have no relation with student's grades logically speaking. eg:address.Thats how you know it worked well.
