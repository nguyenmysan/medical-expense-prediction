# Medical Expense Prediction Using BMI and Age

## Project Overview

This project uses linear regression to examine how BMI and age relate to medical expenses.

A BMI-only model serves as the baseline. Age is then added as a second feature to evaluate whether it improves model fit.

The two-feature model is implemented using:

- **Normal Equation:** Calculates regression coefficients directly using matrix operations.
- **Gradient Descent:** Iteratively updates coefficients to minimize mean squared error.

The project compares all three models using MSE, RMSE, MAE, and R².

## Project Structure

```text
medical-expense-prediction/
├── data/
│   └── insurance-premium-prediction/
│       └── insurance.csv
├── linear-regression/
│   └── Original lab scripts, notebook, and supporting files
├── reports/
│   └── assignment_results.csv
├── memo.md
├── two_feature.py
├── README.md
└── .gitignore
```

| File or Folder | Description |
|---|---|
| `two_feature.py` | Main script for training, evaluating, and comparing the models |
| `data/insurance-premium-prediction/insurance.csv` | Dataset used by the main script |
| `reports/assignment_results.csv` | Exported coefficients and evaluation metrics |
| `memo.md` | Discussion of the results and limitations |
| `linear-regression/` | Original lab materials and supporting implementations |

## Dataset

The dataset contains **1,338 observations**.

The main variables used are:

| Variable | Description | Role |
|---|---|---|
| `bmi` | Body Mass Index | Predictor |
| `age` | Age in years | Predictor |
| `expenses` | Medical expenses | Target |

## Methods

### BMI-Only Baseline

The baseline predicts medical expenses using BMI:

```text
Predicted Medical Expenses = w0 + w1(BMI)
```

The coefficients are calculated using the Normal Equation.

### Normal Equation: BMI + Age

The two-feature model includes both BMI and age:

```text
Predicted Medical Expenses = w0 + w1(BMI) + w2(Age)
```

The script solves the normal-equation linear system using `numpy.linalg.solve`.

### Gradient Descent: BMI + Age

BMI and age are standardized before optimization.

The script uses:

- Learning rate: `0.1`
- Iterations: `10,000`
- Initial coefficients: zeros

After training, the coefficients are converted back to the original feature units so they can be compared with the Normal Equation results.

## Model Evaluation

The models are trained and evaluated on the **full dataset**. The current script does not split the data into training and testing sets.

The reported metrics describe fit to the observed data rather than prediction performance on unseen data.

The evaluation metrics are:

- **MSE:** Average squared prediction error.
- **RMSE:** Prediction error measured in the same units as medical expenses.
- **MAE:** Average absolute prediction error.
- **R²:** Proportion of variation in medical expenses explained by the model.

Lower MSE, RMSE, and MAE indicate smaller errors. Higher R² indicates better model fit.

## How to Run

These instructions support both **macOS and Windows**. Install Python 3 and Git before starting.

The main script requires **NumPy**. The dataset is included in the repository.

### 1. Clone the Repository

Open Terminal on macOS or Command Prompt on Windows, then run:

```bash
git clone https://github.com/nguyenmysan/medical-expense-prediction.git
cd medical-expense-prediction
```

### 2. Install NumPy

**macOS:**

```bash
python3 -m pip install numpy
```

**Windows:**

```bat
py -m pip install numpy
```

### 3. Run the Model

Run the script from the project root directory.

**macOS:**

```bash
python3 two_feature.py
```

**Windows:**

```bat
py two_feature.py
```

If Windows does not recognize `py`, replace it with `python` in both commands, provided `python` runs Python 3.

### 4. View the Results

The script will:

- Load all 1,338 observations.
- Train the BMI-only baseline.
- Train the BMI + age model using the Normal Equation.
- Train the BMI + age model using Gradient Descent.
- Display the coefficients, evaluation metrics, and comparison table.
- Save the results to `reports/assignment_results.csv`.

The script creates the `reports/` folder if needed and overwrites the results CSV when run again.

## Results

| Model | Intercept | BMI Coefficient | Age Coefficient | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| BMI-only Baseline | 1,178.18 | 394.33 | 0.00 | 140,764,214.67 | 11,864.41 | 9,172.30 | 0.0394 |
| Normal Equation: BMI + Age | -6,437.35 | 333.39 | 241.90 | 129,359,773.29 | 11,373.64 | 9,032.28 | 0.1173 |
| Gradient Descent: BMI + Age | -6,437.35 | 333.39 | 241.90 | 129,359,773.29 | 11,373.64 | 9,032.28 | 0.1173 |

Values are rounded for display. Full numerical results are exported to the CSV file.

Adding age increases R² from approximately **0.0394 to 0.1173**, an improvement of approximately **7.78 percentage points**, calculated using the unrounded values.

MSE, RMSE, and MAE also decrease.

The Normal Equation and Gradient Descent produce nearly identical coefficients and metrics, showing that Gradient Descent converges to approximately the same least-squares solution.

## Coefficient Interpretation

The estimated two-feature equation is:

```text
Predicted Medical Expenses
= -6437.35 + 333.39(BMI) + 241.90(Age)
```

- Holding age constant, a one-unit increase in BMI is associated with approximately **$333.39** higher predicted medical expenses.
- Holding BMI constant, a one-year increase in age is associated with approximately **$241.90** higher predicted medical expenses.

These coefficients describe associations in the dataset and do not establish causation.

## Limitations and Future Work

The two-feature model explains approximately **11.73%** of the variation in medical expenses. Most variation remains unexplained.

The current metrics are calculated on the data used for training, so they do not establish performance on new observations.

Future improvements include:

- Evaluating the models on a separate test set.
- Fitting standardization on the training set when implementing a train/test split.
- Exploring additional predictors, such as smoking status.
- Comparing alternative models.

This project demonstrates linear regression methods and their interpretation. Further validation is needed before using the model for individual medical expense predictions.
