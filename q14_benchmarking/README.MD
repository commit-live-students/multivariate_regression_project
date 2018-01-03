###Comparison to decide model
write function that tries every possible combination of strategies you have learnt so far
Compile your results in dataframe named complete_stats.
 your complete stats will have following components
 complete_stats = pd.concat([linear model,linear model with feature selection,chain stats with features selection
 ,stats_chain,stats_lasso,stats_lasso_ft,stats_ridge,stats_ridge_ft])
Here, ft will mean feature selction performed.
 your signature will look like this
 complete_stats  = create_stats(x_train, x_test, y_train, y_test,enc = "labelencoder"):

Having said that we have coded for 2 encoders, we will test it via argument enc.
This will serve as identifier for strategy to us
more of this will be seen in next assignment
