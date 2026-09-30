# Two-Feature Linear Regression 
Adding age improved the R² of the linear regression model from approximately 0.0394 to 0.1173. Using the unrounded results, this represents an increase of approximately 0.0778, or 7.78 percentage points. Including age therefore helps the model explain more variation in medical expenses compared with using BMI alone.

The Normal Equation and Gradient Descent produced nearly identical coefficients. Both methods estimated an intercept of -6437.35, a BMI coefficient of 333.39, and an age coefficient of 241.90. These results agree because both methods minimize the same squared error objective. Standardizing the features helped Gradient Descent converge, and its coefficients were converted back to the original feature units for comparison.

Holding BMI constant, a one-year increase in age is associated with approximately $241.90 higher predicted medical expenses. Holding age constant, a one-unit increase in BMI is associated with approximately $333.39 higher predicted medical expenses. These coefficients describe associations in the dataset and should not be interpreted as causal effects.

Although adding age improved model fit, the two-feature model explains only approximately 11.73% of the variation in medical expenses. Most variation remains unexplained. The RMSE of $11,373.64 also indicates that prediction errors can be substantial. Additional predictors, such as smoking status, may improve the model.

The reported metrics were calculated on the same dataset used to train the models. Therefore, they describe fit to the observed data rather than prediction performance on unseen data. Before practical use, the model should be evaluated on a separate test set and compared with alternative models.