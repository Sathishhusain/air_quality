import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the air quality dataset
file_path = "air_quality_data.csv"
df = pd.read_csv(file_path)

# Check for missing values
df.fillna(df.mean(), inplace=True)

# Splitting data into features and target variable
X = df.drop(columns=['AQI'])  # Features (Pollutants & Environmental Factors)
y = df['AQI']  # Target (Air Quality Index)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the air quality prediction model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Save predictions to an Excel file
df_test = X_test.copy()
df_test['Actual AQI'] = y_test
df_test['Predicted AQI'] = y_pred
df_test.to_excel("air_quality_predictions.xlsx", index=False)

print("Prediction results saved to air_quality_predictions.xlsx")

# Visualization: Predicted vs Actual AQI
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted Air Quality Index (AQI)")
plt.savefig("air_quality_plot.png")
plt.show()
