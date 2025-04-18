import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File name
file_name = "dataset.csv"

# Load the dataset
data = pd.read_csv(file_name)

# Create a directory to save results
results_dir = "EDA_Results"
os.makedirs(results_dir, exist_ok=True)

# Step 1: Basic Information
# Save dataset info
with open(os.path.join(results_dir, "data_info.txt"), "w") as f:
    f.write("Dataset Information:\n")
    data.info(buf=f)

# Save first few rows of the dataset
data.head().to_csv(os.path.join(results_dir, "data_head.csv"), index=False)

# Save summary statistics
summary = data.describe()
summary.to_csv(os.path.join(results_dir, "summary_statistics.csv"))

# Step 2: Checking Missing Values
missing_values = data.isnull().sum()
missing_values.to_csv(os.path.join(results_dir, "missing_values.csv"))

# Step 3: Checking Duplicates
duplicates = data.duplicated().sum()
with open(os.path.join(results_dir, "duplicates.txt"), "w") as f:
    f.write(f"Number of duplicate rows: {duplicates}\n")

# Step 4: Distribution of Numeric Features
numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(results_dir, f"{column}_distribution.png"))
    plt.close()

# Step 5: Boxplots for Outlier Detection
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column], color="orange")
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.savefig(os.path.join(results_dir, f"{column}_boxplot.png"))
    plt.close()

# Step 6: Correlation Analysis
# Select only numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=["float64", "int64"])
correlation_matrix = numeric_data.corr()

# Save correlation matrix
correlation_matrix.to_csv(os.path.join(results_dir, "correlation_matrix.csv"))

# Plot heatmap of correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"))
plt.close()

# Step 7: Countplots for Categorical Features
categorical_columns = data.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=data[column], order=data[column].value_counts().index, palette="viridis")
    plt.title(f"Countplot of {column}")
    plt.ylabel(column)
    plt.xlabel("Count")
    plt.savefig(os.path.join(results_dir, f"{column}_countplot.png"))
    plt.close()

print("EDA completed. Results saved in the 'EDA_Results' folder.")
