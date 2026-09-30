# Medical Expense Prediction Using BMI and Age

### MSBA 265 – Linear Regression Project

## Project Overview

This project examines how **BMI and age** can be used to predict **medical expenses** using linear regression.

The project begins with a baseline model that uses only BMI as a predictor. Age is then added as a second feature to determine whether including additional information improves prediction performance.

Two methods are used to train the two-feature model:

- **Normal Equation** – calculates the regression coefficients analytically
- **Gradient Descent** – iteratively updates the model parameters to minimize prediction error

The two-feature models are compared with the BMI-only baseline using MSE, RMSE, MAE, and R².

---

## Project Structure

```text
medical-expense-prediction/
│
├── data/
│   └── insurance-premium-prediction/
│       └── [dataset files]
│
├── linear-regression/
│   └── [linear regression files]
│
├── reports/
│   └── assignment_results.csv
│
├── memo.md
├── two_feature.py
├── README.md
└── .gitignore
```

### File Descriptions

- **`data/`** – Contains the dataset used for the medical expense prediction analysis.
- **`linear-regression/`** – Contains files related to the original linear regression implementation.
- **`reports/assignment_results.csv`** – Contains the model coefficients and evaluation metrics for the three models.
- **`two_feature.py`** – Main Python script that implements and evaluates the BMI-only and BMI + age linear regression models.
- **`memo.md`** – Written memo summarizing the methodology, results, and key findings.
- **`README.md`** – Provides an overview of the project, methodology, instructions, and results.
- **`.gitignore`** – Specifies files and folders that Git should ignore.

---

## Dataset

The dataset contains **1,338 observations** with information about individuals and their medical expenses.

The main variables used in this project are:

- **BMI** – Body Mass Index
- **Age** – Age of the individual
- **Charges** – Medical expenses and the target variable

The goal is to predict medical charges using BMI and age.

---

## Project Objectives

The main objectives of this project are to:

1. Build a baseline linear regression model using BMI.
2. Add age as a second feature.
3. Implement the two-feature model using the Normal Equation.
4. Implement the same model using Gradient Descent.
5. Compare the results of the two training methods.
6. Evaluate whether adding age improves prediction performance.

---

## Methods

### BMI-Only Baseline

The baseline model uses BMI as the only predictor:

```text
Medical Charges = w0 + w1(BMI)
```

This model provides a reference point for evaluating whether adding age improves prediction performance.

### Normal Equation

The Normal Equation calculates the optimal regression coefficients directly using matrix operations.

For the two-feature model:

```text
Medical Charges = w0 + w1(BMI) + w2(Age)
```

This analytical method calculates the model parameters directly without requiring iterative optimization.

### Gradient Descent

Gradient Descent estimates the model parameters by repeatedly updating the coefficients to minimize prediction error.

BMI and age are standardized during optimization to improve convergence. The final coefficients are converted back to the original feature scale so they can be directly compared with the Normal Equation results.

---

## Model Evaluation

The models are evaluated using four regression metrics:

- **MSE (Mean Squared Error)** – measures the average squared prediction error
- **RMSE (Root Mean Squared Error)** – measures prediction error in the same unit as medical charges
- **MAE (Mean Absolute Error)** – measures the average absolute difference between predicted and actual charges
- **R² (Coefficient of Determination)** – measures the proportion of variation in medical charges explained by the model

Lower MSE, RMSE, and MAE values indicate better prediction performance, while a higher R² indicates that the model explains more variation in the target variable.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/nguyenmysan/medical-expense-prediction.git
```

Move into the project directory:

```bash
cd medical-expense-prediction
```

### 2. Install Required Packages

Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn
```

### 3. Run the Model

Run the main Python script from the project root directory:

```bash
python two_feature.py
```

If your system uses `python3`, run:

```bash
python3 two_feature.py
```

The script will:

- Load the medical expense dataset
- Prepare BMI and age as model features
- Split the data into training and testing sets
- Train the BMI-only baseline model
- Train the BMI + age model using the Normal Equation
- Train the BMI + age model using Gradient Descent
- Generate predictions
- Calculate MSE, RMSE, MAE, and R²
- Save the model results for comparison

---

## Expected Output

After running the model, the results should be approximately:

| Model | w0 | w1 (BMI) | w2 (Age) | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| BMI-only Baseline | 1178.18 | 394.33 | 0.00 | 140,764,214.67 | 11,864.41 | 9,172.30 | 0.0394 |
| Normal Equation (BMI + Age) | -6437.35 | 333.39 | 241.90 | 129,359,773.29 | 11,373.64 | 9,032.28 | 0.1173 |
| Gradient Descent (BMI + Age) | -6437.35 | 333.39 | 241.90 | 129,359,773.29 | 11,373.64 | 9,032.28 | 0.1173 |

The complete results with full numerical precision are saved in:

```text
reports/assignment_results.csv
```

Small differences in displayed decimal places may occur depending on formatting.

---

## Results

The BMI-only baseline produces an R² of approximately **0.0394**.

After adding age, the R² increases to approximately **0.1173** for both the Normal Equation and Gradient Descent models.

Prediction errors also decrease:

- **MSE:** 140.76 million → 129.36 million
- **RMSE:** 11,864.41 → 11,373.64
- **MAE:** 9,172.30 → 9,032.28
- **R²:** 0.0394 → 0.1173

These results show that adding age improves the model compared with using BMI alone.

The Normal Equation and Gradient Descent produce nearly identical coefficients and evaluation metrics. This demonstrates that Gradient Descent successfully converges to approximately the same solution as the analytical Normal Equation.

---

## Model Coefficients

For the BMI + age model, the estimated regression equation is:

```text
Predicted Medical Charges
= -6437.35 + 333.39(BMI) + 241.90(Age)
```

### Interpretation

Holding age constant, a one-unit increase in BMI is associated with an increase of approximately **$333.39** in predicted medical charges.

Holding BMI constant, a one-year increase in age is associated with an increase of approximately **$241.90** in predicted medical charges.

These coefficients represent associations within the linear regression model and should not be interpreted as causal effects.

---

## Key Takeaway

Adding **age** to the BMI-only model improves prediction performance.

The R² increases from approximately **0.0394 to 0.1173**, while MSE, RMSE, and MAE decrease. This indicates that BMI and age together provide more predictive information than BMI alone.

However, the R² remains relatively low, suggesting that BMI and age explain only a limited portion of the variation in medical expenses. Additional variables would likely be necessary to build a more accurate prediction model.

