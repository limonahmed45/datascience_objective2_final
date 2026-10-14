import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(
    "FIFA2026_regression_2_1_match_goal_difference.csv"
)

print("Dataset loaded successfully")

# Show first 5 rows
print("\nFIRST 5 ROWS")
print(df.head().to_string())

# Check dataset size
print("\nDATASET SHAPE")
print(df.shape)

# Check column names
print("\nCOLUMN NAMES")
print(df.columns)
# --------------------------------------------------
# SELECT INPUT VARIABLES (X) AND TARGET (y)
# --------------------------------------------------

features = [
    "FIFA_Rank_Strength_Diff",
    "PPG_Diff",
    "GF_per_Game_Diff",
    "GA_per_Game_Diff",
    "Clean_Sheet_Rate_Diff",
    "Last2_PPG_Diff",
    "Last2_GD_per_Game_Diff",
    "Stage_Code"
]

X = df[features]

y = df["Goal_Difference"]

print("\nX SHAPE:")
print(X.shape)

print("\ny SHAPE:")
print(y.shape)

print("\nFIRST 5 ROWS OF X:")
print(X.head().to_string())

print("\nFIRST 5 VALUES OF y:")
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING AND TESTING DATA")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# --------------------------------------------------
# CREATE AND TRAIN LINEAR REGRESSION MODEL
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nFIRST 5 PREDICTIONS")
print(y_pred[:5])


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print("\nMODEL EVALUATION")
print("R-squared:", round(r2, 3))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# --------------------------------------------------
# MODEL COEFFICIENTS
# --------------------------------------------------

print("\nMODEL INTERCEPT")
print(round(model.intercept_, 4))

print("\nMODEL COEFFICIENTS")

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", round(coefficient, 4))


    import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# ACTUAL VS PREDICTED GRAPH - REGRESSION 2.1
# --------------------------------------------------

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

minimum_value = min(y_test.min(), y_pred.min())
maximum_value = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value]
)

plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.title("Actual vs Predicted Goal Difference - Regression 2.1")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "regression_2_1_actual_vs_predicted.png"
)


# --------------------------------------------------
# SAVE REGRESSION 2.1 PREDICTIONS TO CSV
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual_Goal_Difference": y_test.values,
    "Predicted_Goal_Difference": y_pred
})

prediction_results.to_csv(
    "regression_2_1_predictions.csv",
    index=False
)

print("\nCSV FILE SAVED SUCCESSFULLY")
print(
    "Location:",
    os.path.abspath("regression_2_1_predictions.csv")
)

plt.show()