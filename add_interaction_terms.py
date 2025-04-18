import pandas as pd

# Load your dataset
df = pd.read_csv("dataset.csv")  # Replace with the actual dataset file path

# Add interaction terms
df['CO2_Drought'] = df['High_CO2'] * df['Drought']
df['Temp_Precip'] = df['High_Temperature'] * df['Precipitation']
df['ExtremeWeather_Adapt'] = df['Extreme_Weather'] * df['Adaptation_Strategies']
df['CO2_Temp'] = df['High_CO2'] * df['High_Temperature']

# Save the updated dataset
df.to_csv("dataset_with_interactions.csv", index=False)
print("New dataset with interaction terms saved as 'dataset_with_interactions.csv'")
