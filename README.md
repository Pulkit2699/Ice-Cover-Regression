# Ice-Cover-Regression

Ed Hopkins over at the Wisconsin State Climatology OfficeLinks to an external site. maintains a listing of the dates and durations of full-freeze ice covers on our dear Madison lakes, Mendota and Monona, found hereLinks to an external site.. This data goes back to the mid-1800s, thanks to handwritten records in some very old log books (they're behind glass in Ed's office over in AOS).

Analyzing this data is an interesting exploration in the changes in our local climate over time, so let's use some machine learning techniques to take a look at the data.

Functions:
get_dataset() — takes no arguments and returns the data as described below in an n-by-2 array
print_stats(dataset) — takes the dataset as produced by the previous function and prints several statistics about the data; does not return anything
regression(beta_0, beta_1) — calculates and returns the mean squared error on the dataset given fixed betas
gradient_descent(beta_0, beta_1) — performs a single step of gradient descent on the MSE and returns the derivative values as a tuple
iterate_gradient(T, eta) — performs T iterations of gradient descent starting at LaTeX: (\beta_0, \beta_1) = (0,0)( β 0 , β 1 ) = ( 0 , 0 ) with the given parameter and prints the results; does not return anything
compute_betas() — using the closed-form solution, calculates and returns the values of LaTeX: \beta_0β 0 and LaTeX: \beta_1β 1 and the corresponding MSE as a three-element tuple
predict(year) — using the closed-form solution betas, return the predicted number of ice days for that year
iterate_normalized(T, eta) — normalizes the data before performing gradient descent, prints results as in function 5
sgd(T, eta) — performs stochastic gradient descent, prints results as in function 5
