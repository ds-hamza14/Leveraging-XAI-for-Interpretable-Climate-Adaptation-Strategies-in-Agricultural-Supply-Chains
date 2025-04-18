import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Feature Engineering

# Create interaction terms
df['Temperature_Precipitation_Interaction'] = df['Average_Temperature_C'] * df['Total_Precipitation_mm']
df['CO2_Temperature_Interaction'] = df['CO2_Emissions_MT'] * df['Average_Temperature_C']

# Create bins for temperature and precipitation
temperature_bins = [-np.inf, 15, 25, np.inf]
temperature_labels = ['Low', 'Moderate', 'High']
df['Temperature_Bin'] = pd.cut(df['Average_Temperature_C'], bins=temperature_bins, labels=temperature_labels)

precipitation_bins = [-np.inf, 500, 1000, np.inf]
precipitation_labels = ['Dry', 'Moderate', 'Wet']
df['Precipitation_Bin'] = pd.cut(df['Total_Precipitation_mm'], bins=precipitation_bins, labels=precipitation_labels)

# Ensure that the 'Temperature_Bin' and 'Precipitation_Bin' columns are created successfully
print("Temperature_Bin column after binning:")
print(df[['Average_Temperature_C', 'Temperature_Bin']].head())

print("\nPrecipitation_Bin column after binning:")
print(df[['Total_Precipitation_mm', 'Precipitation_Bin']].head())

# 2. Encode categorical variables (Country, Region, Crop_Type)
categorical_cols = ['Country', 'Region', 'Crop_Type', 'Temperature_Bin', 'Precipitation_Bin']

encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# 3. Scaling and Normalization
# Standardize numerical columns
scaler = StandardScaler()
numerical_cols = ['Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT', 
                  'Crop_Yield_MT_per_HA', 'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
                  'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index', 'Economic_Impact_Million_USD']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 4. Handle Imbalanced Classes (Adaptation_Strategies)
# Apply SMOTE for oversampling the minority class in the classification target
smote = SMOTE(sampling_strategy='auto', random_state=42)
X = df.drop(columns=['Adaptation_Strategies'])
y = df['Adaptation_Strategies']
X_res, y_res = smote.fit_resample(X, y)
df_resampled = pd.concat([X_res, y_res], axis=1)

# 5. Feature Selection: Remove features with low correlation or high multicollinearity
# Calculate the correlation matrix (exclude non-numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Calculate VIF for multicollinearity
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
high_vif_vars = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()

# Drop features with low correlation or high VIF
df_final = df.drop(columns=high_vif_vars)

# 6. Recursive Feature Elimination (RFE) with RandomForest to select important features
from sklearn.ensemble import RandomForestClassifier

# Using RandomForest to select important features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(rf, n_features_to_select=10)
selector = selector.fit(X_res, y_res)

# Select top features
selected_columns = X.columns[selector.support_]
df_final_rfe = df_resampled[selected_columns.tolist() + ['Adaptation_Strategies']]

# Save the final preprocessed dataset
df_final_rfe.to_csv('preprocessed_final_dataset.csv', index=False)

# Show selected features and the final dataset
selected_columns, df_final_rfe.head()
