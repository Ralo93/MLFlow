
import pandas as pd
import mlflow

# Load the unseen dataset
unseen_data = pd.read_csv(r'../data/test_values.csv')

# Load the saved pipeline (preprocessor + XGBoost model)
logged_model = 'runs:/<run_id>/model'
loaded_pipeline = mlflow.sklearn.load_model(logged_model)

# Perform predictions on the unseen dataset
predictions = loaded_pipeline.predict(unseen_data)
predictions = predictions.replace({0: 1, 1: 2, 2: 3})

# Print the predictions or save them to a file
print(predictions)

# If you want to save the predictions to a file
output_df = pd.DataFrame(predictions, columns=["Predicted damage_grade"])
output_df.to_csv(r'../outputs/predictions.csv', index=False)
