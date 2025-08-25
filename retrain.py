import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("xgboost_predictable_stock.csv")
#df = pd.read_csv("xgboost_stock_extended.csv")

# Drop missing/null values if any
df.dropna(inplace=True)

# Recalculate VWAP
df["VWAP"] = (df["High"] + df["Low"] + df["Close"]) / 3

# Features and target
features = ["Open", "High", "Low", "Close", "Volume"]
target = "VWAP"

X = df[features]
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Model with optimized parameters
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    reg_lambda=1,
    reg_alpha=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
def evaluate(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # Compatible RMSE
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {dataset_name} Evaluation")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Predict and evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

evaluate(y_train, y_pred_train, "Train")
evaluate(y_test, y_pred_test, "Test")

# Save the model
joblib.dump(model, "xgboost_model.json")
print("\n✅ New model trained & saved as xgboost_model.pkl")

# Plot feature importance
xgb.plot_importance(model)
plt.title('📊 Feature Importance - Updated XGBoost VWAP Model')
plt.tight_layout()
plt.savefig("feature_importance_updated.png")
plt.show()