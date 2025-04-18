import pandas as pd
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Load the dataset used for testing (ensure it's the same dataset)
data = pd.read_csv("final_dataset_after_vif.csv")

# Separate features and target variable
X = data.drop(columns=["Economic_Impact_Million_USD", "Adaptation_Strategies"])
y = data["Economic_Impact_Million_USD"]

# Preprocessing to handle non-numeric data or missing values
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)  # Replace NaN values with 0

# Select a single sample for comparison
sample_index = 0  # Choose the first sample for this comparison
sample = X.iloc[sample_index:sample_index+1]

# ---- SHAP Analysis ----
print("Starting SHAP Analysis...")
explainer_shap = shap.Explainer(model, X)

# Disable additivity check to avoid precision errors
shap_values = explainer_shap(X, check_additivity=False)

# SHAP summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary_plot.png")
plt.clf()

# SHAP force plot for the single sample
shap.save_html(
    "shap_force_plot.html",
    shap.force_plot(
        base_value=shap_values.base_values[sample_index],
        shap_values=shap_values.values[sample_index],
        features=sample.values[0],
        feature_names=X.columns.tolist()
    )
)
print("SHAP Analysis Completed. Visualizations saved.")

# ---- LIME Analysis ----
print("Starting LIME Analysis...")
explainer_lime = LimeTabularExplainer(
    X.values,
    training_labels=y.values,
    mode="regression",
    feature_names=X.columns.tolist(),
)

lime_explanation = explainer_lime.explain_instance(sample.values[0], model.predict, num_features=5)
lime_explanation.save_to_file("lime_explanation.html")
print("LIME Analysis Completed. Explanation saved.")

# Compare Results
print("\nComparison Completed:")
print("- SHAP feature importance visualization saved as 'shap_summary_plot.png'.")
print("- SHAP individual prediction explanation saved as 'shap_force_plot.html'.")
print("- LIME explanation saved as 'lime_explanation.html'.")
