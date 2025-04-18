import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('final_dataset_after_vif.csv')

# Display the column names to check if they match
print("Columns in the dataset:", df.columns)

# Select the target variable and features
target = 'Economic_Impact_Million_USD'
features = ['Total_Precipitation_mm', 'CO2_Emissions_MT', 'Crop_Yield_MT_per_HA', 
            'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA', 'Fertilizer_Use_KG_per_HA', 
            'Soil_Health_Index', 'Temperature_Precipitation_Interaction', 'CO2_Temperature_Interaction']

X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model from Grid Search
best_rf = grid_search.best_estimator_

# Predictions using the best model
y_pred = best_rf.predict(X_test_scaled)

# Evaluate the model performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the performance metrics
print(f"Random Forest Regressor Performance after tuning:")
print(f"R²: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

# Feature Importance Visualization
feature_importances = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(features)), np.array(features)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest Model')
plt.show()

# Cross-validation score to evaluate model generalization
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {-cv_scores}")
print(f"Average CV MSE: {-cv_scores.mean()}")

# Predictions for extreme conditions
# Simulate extreme conditions by altering Temperature and Precipitation
extreme_conditions = X_test_scaled.copy()
extreme_conditions[:, 0] += 10  # Increase precipitation
extreme_conditions[:, 7] += 5   # Increase temperature interaction

# Predict with extreme conditions
extreme_pred = best_rf.predict(extreme_conditions)

# Print the predictions for extreme conditions
print("Predictions for Extreme Conditions:", extreme_pred)

# Save the model and results for later use
import joblib
joblib.dump(best_rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save results to a CSV file
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Extreme_Predictions': extreme_pred
})

results.to_csv('model_predictions.csv', index=False)

print("Model and predictions saved successfully.")
