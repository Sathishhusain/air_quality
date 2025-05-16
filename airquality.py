import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the air quality data from Excel
file_path = "air_quality_data.xlsx"
df = pd.read_excel(file_path)

# Convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Define features (X) and target variable (y)
features = ["temperature", "humidity", "wind_speed", "PM2.5", "PM10", "NO2", "SO2", "O3"]
X = df[features]
y = df["AQI"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict AQI values
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualizing actual vs predicted AQI values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual AQI", marker="o", linestyle="-")
plt.plot(y_pred, label="Predicted AQI", marker="s", linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.title("Actual vs Predicted AQI Levels")
plt.legend()
plt.grid()
plt.show()
