import lime
import lime.lime_tabular
import pandas as pd
import joblib

# Load the trained model and dataset
model = joblib.load('random_forest_model.pkl')
data = pd.read_csv('final_dataset_after_vif.csv')

# Define features and target
X = data.drop(columns=['Economic_Impact_Million_USD', 'Adaptation_Strategies'])
y = data['Economic_Impact_Million_USD']

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values,
    feature_names=X.columns,
    class_names=['Economic Impact'],
    verbose=True,
    mode='regression'
)

# Select an instance to explain
sample_index = 0  # Choose an index to explain
sample = X.iloc[sample_index].values.reshape(1, -1)

# Generate explanation
exp = explainer.explain_instance(sample.flatten(), model.predict, num_features=10)

# Show explanation
exp.show_in_notebook(show_table=True)
exp.save_to_file('lime_explanation.html')

print("LIME analysis completed. Explanation saved as lime_explanation.html.")
