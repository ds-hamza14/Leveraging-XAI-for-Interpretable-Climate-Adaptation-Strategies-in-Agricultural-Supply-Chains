import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import pearsonr

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_final_dataset.csv')

# Check and clean column names (remove any extra spaces)
df.columns = df.columns.str.strip()

# 1. Exclude non-numeric columns for correlation matrix
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]

# 2. Correlation Matrix: Investigate correlations between independent variables and target variables
corr_matrix = df_numeric.corr()

# Display the correlation matrix (with respect to Crop_Yield_MT_per_HA and Economic_Impact_Million_USD)
print("Correlation Matrix with target variables:")
print(corr_matrix[['Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD']])

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix[['Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD']].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation between independent variables and target variables")
plt.show()

# 3. Partial Correlation: Investigate the relationship between Extreme_Weather_Events and the target while controlling for others
# Ensure the column 'Extreme_Weather_Events' exists
if 'Extreme_Weather_Events' in df.columns:
    # First, create a dataframe excluding 'Extreme_Weather_Events'
    df_no_weather = df.drop(columns=['Extreme_Weather_Events'])

    # Then, run partial correlation between 'Extreme_Weather_Events' and 'Crop_Yield_MT_per_HA' while controlling for others
    X = df_no_weather.drop(columns=['Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD'])
    y = df['Crop_Yield_MT_per_HA']
    X = add_constant(X)

    # Fit the model and obtain residuals
    model = sm.OLS(y, X).fit()
    residuals_yield = model.resid

    # Fit a similar model for Extreme_Weather_Events
    X_weather = df_no_weather.drop(columns=['Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD'])
    X_weather = add_constant(X_weather)

    weather_model = sm.OLS(df['Extreme_Weather_Events'], X_weather).fit()
    residuals_weather = weather_model.resid

    # Now, compute the partial correlation between residuals
    partial_corr, _ = pearsonr(residuals_weather, residuals_yield)

    print(f"Partial correlation between 'Extreme_Weather_Events' and 'Crop_Yield_MT_per_HA' (controlling for other variables): {partial_corr}")
else:
    print("'Extreme_Weather_Events' column not found in the dataset.")

# 4. Multicollinearity: Calculate Variance Inflation Factor (VIF)
# Calculate VIF for all features
X = df.drop(columns=['Economic_Impact_Million_USD', 'Crop_Yield_MT_per_HA', 'Adaptation_Strategies'])  # Exclude target variables
X = add_constant(X)

vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print VIF results
print("\nVariance Inflation Factor (VIF) for features:")
print(vif_data)

# Drop features with high VIF (threshold of 10 or higher) to reduce multicollinearity
high_vif_vars = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
print(f"\nDropping features with VIF > 10: {high_vif_vars}")

df_final = df.drop(columns=high_vif_vars)

# Save the final dataset after handling multicollinearity
df_final.to_csv('final_dataset_after_vif.csv', index=False)

# Display the final cleaned dataset
df_final.head()
