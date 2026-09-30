
import csv
import numpy as np
from pathlib import Path

# Step 1: Load the insurance dataset
DATA_PATH = Path(__file__).parent / "data" / "insurance-premium-prediction" / "insurance.csv"

bmi = []
age = []
expenses = []

with open(DATA_PATH, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        bmi.append(float(row["bmi"]))
        age.append(float(row["age"]))
        expenses.append(float(row["expenses"]))

# Convert lists to NumPy arrays
bmi = np.array(bmi)
age = np.array(age)
expenses = np.array(expenses)

print("Dataset loaded successfully!")
print("Number of observations:", len(expenses))
print("First 5 BMI values:", bmi[:5])
print("First 5 age values:", age[:5])
print("First 5 expenses:", expenses[:5])



# Step 2: Fit the two-feature model using the Normal Equation

# Create the feature matrix with an intercept column
X = np.column_stack((np.ones(len(bmi)), bmi, age))

# Target variable
y = expenses

# Calculate the Normal Equation
XT_X = X.T @ X
XT_y = X.T @ y

# Solve for the model weights
weights_normal = np.linalg.solve(XT_X, XT_y)

# Extract the three weights
w0, w1, w2 = weights_normal

# Make predictions
y_pred_normal = X @ weights_normal

print("\nNormal Equation Results:")
print(f"w0 (Intercept): {w0:.2f}")
print(f"w1 (BMI): {w1:.2f}")
print(f"w2 (Age): {w2:.2f}")


# Step 3: Gradient Descent with two standardized features

# Standardize both BMI and age
def standardize(x):
    mean = float(np.mean(x))
    std = float(np.std(x))

    if std == 0:
        return np.zeros_like(x), mean, 1.0

    return (x - mean) / std, mean, std


bmi_std, bmi_mean, bmi_scale = standardize(bmi)
age_std, age_mean, age_scale = standardize(age)

# Create the standardized feature matrix
X_std = np.column_stack((
    np.ones(len(bmi)),
    bmi_std,
    age_std
))

print("\nStandardization Results:")
print(f"BMI mean: {bmi_mean:.2f}")
print(f"BMI standard deviation: {bmi_scale:.2f}")
print(f"Age mean: {age_mean:.2f}")
print(f"Age standard deviation: {age_scale:.2f}")


# Step 4: Train the model using Gradient Descent

# Initialize the model weights
weights_gd_std = np.zeros(3)

# Set training parameters
learning_rate = 0.1
iterations = 10000
n = len(y)

# Gradient Descent training loop
for i in range(iterations):

    # Make predictions
    predictions = X_std @ weights_gd_std

    # Calculate prediction errors
    errors = predictions - y

    # Calculate the gradient
    gradient = (2 / n) * (X_std.T @ errors)

    # Update the weights
    weights_gd_std = weights_gd_std - learning_rate * gradient

# Convert weights back to original BMI and age units
w1_gd = weights_gd_std[1] / bmi_scale
w2_gd = weights_gd_std[2] / age_scale

w0_gd = (
    weights_gd_std[0]
    - w1_gd * bmi_mean
    - w2_gd * age_mean
)

# Store the original-unit weights
weights_gd = np.array([w0_gd, w1_gd, w2_gd])

# Make predictions using the original features
y_pred_gd = X @ weights_gd

print("\nGradient Descent Results:")
print(f"w0 (Intercept): {w0_gd:.2f}")
print(f"w1 (BMI): {w1_gd:.2f}")
print(f"w2 (Age): {w2_gd:.2f}")


# Step 5: Evaluate both models

# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Root Mean Squared Error
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# R-squared
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_total)


# Calculate metrics for the Normal Equation
normal_mse = mse(y, y_pred_normal)
normal_rmse = rmse(y, y_pred_normal)
normal_mae = mae(y, y_pred_normal)
normal_r2 = r2(y, y_pred_normal)


# Calculate metrics for Gradient Descent
gd_mse = mse(y, y_pred_gd)
gd_rmse = rmse(y, y_pred_gd)
gd_mae = mae(y, y_pred_gd)
gd_r2 = r2(y, y_pred_gd)


# Display the results
print("\nModel Evaluation Results:")

print("\nNormal Equation:")
print(f"MSE: {normal_mse:.2f}")
print(f"RMSE: {normal_rmse:.2f}")
print(f"MAE: {normal_mae:.2f}")
print(f"R²: {normal_r2:.4f}")

print("\nGradient Descent:")
print(f"MSE: {gd_mse:.2f}")
print(f"RMSE: {gd_rmse:.2f}")
print(f"MAE: {gd_mae:.2f}")
print(f"R²: {gd_r2:.4f}")


# Step 6: Compare with the BMI-only baseline

# Create the baseline feature matrix
X_baseline = np.column_stack((
    np.ones(len(bmi)),
    bmi
))

# Fit the BMI-only model using the Normal Equation
baseline_weights = np.linalg.solve(
    X_baseline.T @ X_baseline,
    X_baseline.T @ y
)

# Make baseline predictions
y_pred_baseline = X_baseline @ baseline_weights

# Calculate baseline metrics
baseline_mse = mse(y, y_pred_baseline)
baseline_rmse = rmse(y, y_pred_baseline)
baseline_mae = mae(y, y_pred_baseline)
baseline_r2 = r2(y, y_pred_baseline)

# Calculate the improvement in R-squared
r2_improvement = normal_r2 - baseline_r2

print("\nBMI-only Baseline Results:")
print(f"w0 (Intercept): {baseline_weights[0]:.2f}")
print(f"w1 (BMI): {baseline_weights[1]:.2f}")
print(f"MSE: {baseline_mse:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"MAE: {baseline_mae:.2f}")
print(f"R²: {baseline_r2:.4f}")

print("\nR² Improvement:")
print(f"BMI-only model: {baseline_r2:.4f}")
print(f"BMI + Age model: {normal_r2:.4f}")
print(f"Improvement: {r2_improvement:.4f}")


# Step 7: Create a comparison table and export results to CSV

# Create a list of model results
results = [
    {
        "Model": "BMI-only Baseline",
        "w0": baseline_weights[0],
        "w1 (BMI)": baseline_weights[1],
        "w2 (Age)": 0.0,
        "MSE": baseline_mse,
        "RMSE": baseline_rmse,
        "MAE": baseline_mae,
        "R2": baseline_r2
    },
    {
        "Model": "Normal Equation (BMI + Age)",
        "w0": weights_normal[0],
        "w1 (BMI)": weights_normal[1],
        "w2 (Age)": weights_normal[2],
        "MSE": normal_mse,
        "RMSE": normal_rmse,
        "MAE": normal_mae,
        "R2": normal_r2
    },
    {
        "Model": "Gradient Descent (BMI + Age)",
        "w0": weights_gd[0],
        "w1 (BMI)": weights_gd[1],
        "w2 (Age)": weights_gd[2],
        "MSE": gd_mse,
        "RMSE": gd_rmse,
        "MAE": gd_mae,
        "R2": gd_r2
    }
]

# Print a readable comparison table
print("\nModel Comparison Table")
print("-" * 125)

header = (
    f"{'Model':<32}"
    f"{'w0':>12}"
    f"{'w1 (BMI)':>12}"
    f"{'w2 (Age)':>12}"
    f"{'MSE':>17}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'R2':>10}"
)

print(header)
print("-" * 125)

for result in results:
    print(
        f"{result['Model']:<32}"
        f"{result['w0']:>12.2f}"
        f"{result['w1 (BMI)']:>12.2f}"
        f"{result['w2 (Age)']:>12.2f}"
        f"{result['MSE']:>17.2f}"
        f"{result['RMSE']:>12.2f}"
        f"{result['MAE']:>12.2f}"
        f"{result['R2']:>10.4f}"
    )

# Create the reports folder
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Export results to CSV
CSV_PATH = REPORTS_DIR / "assignment_results.csv"

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults exported successfully to: {CSV_PATH}")
