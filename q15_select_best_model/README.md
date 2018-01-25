###Complete function that returns statistics
Write function to return statistics of all tested paths in this projects ordered by r2 and rmse
Feel free to plot few things to check how the visuals look.

This function will call ohe encoding too for that accuracies.
Benchmarking is usually given with dataset if it is experimental one like this one is.
in this case, this data can be found  http://www3.dsi.uminho.pt/pcortez/student.pdf
you should find listed RMSE regression results if you look into this paper.


function signature should look like this
model_plot = complete_build(x_train, x_test, y_train, y_test)

Just to be clear, These are the demonstrations of strategies to be used in ML while solving real life problem.
This is not completely tweaked and developed model.
I encourage you to try out some parameters and note the fluctuations in results.
This should get you pretty good intuition of what to use.

keep plotting residuals using previous assignments and save plots to images to draw conclusions. 
Remember, your residual plot should be randomly scattered along horizontal line.
if it displays some kind of curved pattern, it is not linear fit
If it fits for a while and then scatters too much , may be, its good for range of x-values.

Great job ML aspirants!


###For further modification
I suggest remove irrelevant features manually before feeding to k_best features.
Most likely should be address/absences etc, take a look at correlation plot for this.
You may want to change parameters to check appropriate value.
Tweak parameters like alpha for lasso and ridge.

Signing Off!!
