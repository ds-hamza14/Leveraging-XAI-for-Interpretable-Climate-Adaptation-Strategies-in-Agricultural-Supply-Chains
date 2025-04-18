import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and dataset
model = joblib.load('random_forest_model.pkl')
data = pd.read_csv('final_dataset_after_vif.csv')

# Define features and target
X = data.drop(columns=['Economic_Impact_Million_USD', 'Adaptation_Strategies'])
y = data['Economic_Impact_Million_USD']

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot (feature importance)
plt.title("SHAP Feature Importance")
shap.summary_plot(shap_values, X)

# Individual prediction explanation
sample_index = 0  # Choose a sample index to explain
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index, :],
    X.iloc[sample_index, :],
    matplotlib=True
)

# Save the summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig('shap_summary_plot.png')

print("SHAP analysis completed. Feature importance and individual prediction explanations saved.")
