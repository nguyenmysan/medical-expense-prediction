# Two-Feature Linear Regression – Written Memo

Adding age improved the R² of the linear regression model from 0.0394 to 0.1173, an increase of approximately 0.0778, or 7.78 percentage points. This means that including age helps the model explain more variation in insurance expenses compared to using BMI alone.

The Normal Equation and Gradient Descent produced nearly identical weights. Both methods estimated an intercept of -6437.35, a BMI coefficient of 333.39, and an age coefficient of 241.90. These results agree because both methods minimize the same squared error objective, and Gradient Descent converges toward the optimal solution when the learning rate is appropriate.

The positive age coefficient of 241.90 indicates that insurance expenses are predicted to increase by approximately $241.90 for each additional year of age when BMI remains constant. This suggests that age is positively associated with insurance expenses in this dataset.

The two-feature model has limited predictive power, with an R² of 0.1173. This means the model explains approximately 11.73% of the variation in insurance expenses, leaving most of the variation unexplained. The RMSE of $11,373.64 also indicates that prediction errors can be substantial. Important factors such as smoking status, medical conditions, and other relevant variables may help improve the model. Before deployment, the model should be evaluated on unseen data and compared with more advanced approaches to determine whether its predictions are reliable.
