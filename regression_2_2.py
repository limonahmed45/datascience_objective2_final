import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# --------------------------------------------------
# REGRESSION 2.2
# Predict goals scored by one team in one match
# --------------------------------------------------


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    "FIFA2026_regression_2_2_team_goals.csv"
)

print("Dataset loaded successfully")


# --------------------------------------------------
# 2. CHECK DATASET
# --------------------------------------------------

print("\nFIRST 5 ROWS")
print(df.head().to_string())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)


# --------------------------------------------------
# 3. SELECT THE 8 PREDICTORS
# --------------------------------------------------

features = [
    "FIFA_Rank_Strength_Diff",
    "Team_Pre_GF_per_Game",
    "Team_Pre_PPG",
    "Team_Pre_Clean_Sheet_Rate",
    "Team_Pre_Last2_GF_per_Game",
    "Opponent_Pre_GA_per_Game",
    "Opponent_Pre_PPG",
    "Stage_Code"
]


# X = predictors
X = df[features]

# y = target
y = df["Goals_Scored"]


print("\nX SHAPE")
print(X.shape)

print("\ny SHAPE")
print(y.shape)


# --------------------------------------------------
# 4. SPLIT DATA BY MATCH ID
# --------------------------------------------------

# Get the 104 unique match IDs
unique_matches = df["Match_ID"].unique()

print("\nTOTAL UNIQUE MATCHES")
print(len(unique_matches))


# Split matches into training and testing groups
train_matches, test_matches = train_test_split(
    unique_matches,
    test_size=0.20,
    random_state=42
)


# Keep both team rows from the same match together
train_df = df[
    df["Match_ID"].isin(train_matches)
]

test_df = df[
    df["Match_ID"].isin(test_matches)
]


# Create training X and y
X_train = train_df[features]
y_train = train_df["Goals_Scored"]


# Create testing X and y
X_test = test_df[features]
y_test = test_df["Goals_Scored"]


print("\nTRAINING AND TESTING DATA")

print("Training matches:", len(train_matches))
print("Testing matches:", len(test_matches))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# --------------------------------------------------
# 5. CREATE THE LINEAR REGRESSION MODEL
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# 6. TRAIN THE MODEL
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)

print("\nModel trained successfully")


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(
    X_test
)

print("\nFIRST 10 PREDICTIONS")
print(
    np.round(
        y_pred[:10],
        2
    )
)


# --------------------------------------------------
# 8. ACTUAL VS PREDICTED GOALS
# --------------------------------------------------

results = pd.DataFrame({
    "Actual_Goals": y_test.values,
    "Predicted_Goals": y_pred
})


print("\nACTUAL VS PREDICTED")

print(
    results.head(10)
    .round(2)
    .to_string(index=False)
)


# --------------------------------------------------
# 9. TEST SET MODEL EVALUATION
# --------------------------------------------------

test_r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


print("\nTEST SET MODEL EVALUATION")

print(
    "R-squared:",
    round(test_r2, 3)
)

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)


# --------------------------------------------------
# 10. TRAINING SET R-SQUARED
# --------------------------------------------------

y_train_pred = model.predict(
    X_train
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)


print("\nTRAINING R-SQUARED")

print(
    round(
        train_r2,
        3
    )
)


# --------------------------------------------------
# 11. MODEL INTERCEPT
# --------------------------------------------------

print("\nMODEL INTERCEPT")

print(
    round(
        model.intercept_,
        4
    )
)


# --------------------------------------------------
# 12. MODEL COEFFICIENTS
# --------------------------------------------------

print("\nMODEL COEFFICIENTS")

for feature, coefficient in zip(
    features,
    model.coef_
):
    print(
        feature,
        ":",
        round(
            coefficient,
            4
        )
    )


# --------------------------------------------------
# 13. FINAL INTERPRETATION
# --------------------------------------------------

print("\nFINAL INTERPRETATION")

print(
    "The model uses eight pre-match variables "
    "to predict the number of goals scored by a team."
)

print(
    "The train-test split was performed using Match_ID "
    "so both teams from the same match remain together."
)

print(
    "R-squared shows how much variation in team goals "
    "is explained by the regression model."
)

print(
    "MAE shows the average absolute difference between "
    "the predicted and actual number of goals."
)

print(
    "RMSE is another error measure that gives more "
    "penalty to larger prediction errors."
)

import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 14. ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

# Perfect prediction reference line
minimum_value = min(y_test.min(), y_pred.min())
maximum_value = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value]
)

plt.xlabel("Actual Goals")
plt.ylabel("Predicted Goals")
plt.title("Actual vs Predicted Goals - Regression 2.2")
plt.grid(True)
plt.tight_layout()

# Save graph
plt.savefig("regression_2_2_actual_vs_predicted.png")


# --------------------------------------------------
# 15. SAVE PREDICTIONS TO CSV
# --------------------------------------------------

prediction_results = test_df[
    ["Match_ID", "Team", "Opponent", "Goals_Scored"]
].copy()

prediction_results["Predicted_Goals"] = y_pred

prediction_results = prediction_results.rename(
    columns={
        "Goals_Scored": "Actual_Goals"
    }
)

# Save CSV in your current F:\Fifa_2026 folder
csv_file = "regression_2_2_predictions.csv"

prediction_results.to_csv(
    csv_file,
    index=False
)

print("\nCSV FILE SAVED SUCCESSFULLY")
print("Location:", os.path.abspath(csv_file))


plt.show()