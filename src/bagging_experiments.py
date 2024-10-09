import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import BaseNEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


# Set up basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading Data...")
data = pd.read_csv(r'../data/train_clean.csv')
logging.info("Done")

x = data.drop(columns=['Promoted_or_Not'])
y = data['Promoted_or_Not']

logging.info("Starting Preprocessing...")

# Define numerical and categorical columns
numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in x.columns if x[cname].nunique() < 800 and x[cname].dtype == "object"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Define preprocessor pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

base_encoder_columns = ['Division', 'Qualification', 'Channel_of_Recruitment', 'State_Of_Origin',
                        'Foreign_schooled', 'Marital_Status', 'Previous_IntraDepartmental_Movement',
                        'No_of_previous_employers', 'Gender']
base_encoder = Pipeline(steps=[
    ('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))
])

# Combine the transformations using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('base_name', base_encoder, base_encoder_columns),
    ('num', numerical_transformer, numerical_cols)
])

logging.info("Done.")

logging.info("Starting Resampling...")

# Combine features and target into a single DataFrame for resampling
df_train = pd.concat([X_train, y_train], axis=1)

# Identify the majority and minority classes
majority_class = df_train[df_train['Promoted_or_Not'] == 0]
minority_class = df_train[df_train['Promoted_or_Not'] == 1]

# Upsample the minority class
minority_class_upsampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),
    random_state=42
)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([majority_class, minority_class_upsampled])

# Shuffle the dataset
df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target again
X_train_upsampled = df_upsampled.drop('Promoted_or_Not', axis=1)
y_train_upsampled = df_upsampled['Promoted_or_Not']
logging.info("Done.")

logging.info("Starting Model...")

# Create BalancedBaggingClassifier with DecisionTreeClassifier as base estimator
bb_classifier = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

# Model pipeline
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', bb_classifier)  # Changed 'xgboost' to 'classifier'
])

# Function to train the model with hyperparameters
def train_model(params, train_x, train_y, valid_x, valid_y, test_x=None, test_y=None):
    logging.info("Starting Training...")

    # Set up the base estimator with the desired parameters
    base_estimator = DecisionTreeClassifier(
        max_depth=int(params['base_max_depth']),
        min_samples_split=int(params['base_min_samples_split']),
        min_samples_leaf=int(params['base_min_samples_leaf']),
        random_state=42
    )

    # Update the classifier parameters
    rf_pipe.set_params(
        classifier__estimator=base_estimator,
        classifier__n_estimators=params['n_estimators'],
        classifier__max_samples=params.get('max_samples', 1.0),
        classifier__max_features=params.get('max_features', 1.0),
        classifier__bootstrap=params['bootstrap'],
        classifier__bootstrap_features=params['bootstrap_features'],
        classifier__random_state=42

    )

    # Train the model with MLflow tracking
    with mlflow.start_run(nested=True):
        # Fit the pipeline with training data
        rf_pipe.fit(train_x, train_y)

        # Predict on the validation set
        valid_preds = rf_pipe.predict(valid_x)

        # Calculate evaluation metrics
        eval_accuracy = accuracy_score(valid_y, valid_preds)
        eval_precision = precision_score(valid_y, valid_preds, average='macro')
        eval_recall = recall_score(valid_y, valid_preds, average='macro')
        eval_f1_score = f1_score(valid_y, valid_preds, average='macro')

        # Get classification report for all classes
        class_report = classification_report(valid_y, valid_preds, output_dict=True)

        # Log parameters and results to MLflow
        mlflow.log_params(params)
        # Log metrics for each class separately
        for class_label, metrics in class_report.items():
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                mlflow.log_metric(f"precision_class_{class_label}", metrics['precision'])
                mlflow.log_metric(f"recall_class_{class_label}", metrics['recall'])
                mlflow.log_metric(f"f1_score_class_{class_label}", metrics['f1-score'])

        # Log overall metrics
        mlflow.log_metric("macro_f1_score", eval_f1_score)
        mlflow.log_metric("macro_precision", eval_precision)
        mlflow.log_metric("macro_recall", eval_recall)
        mlflow.log_metric("accuracy", eval_accuracy)

        mlflow.set_tag("tag", "evaluation after accuracy")

        # Return the model and loss
        return {"loss": -eval_accuracy, "status": STATUS_OK, "model": rf_pipe}

# Define the objective function for hyperopt
def objective(params):
    result = train_model(
        params,
        train_x=X_train,
        train_y=y_train,
        valid_x=X_test,
        valid_y=y_test,
        test_x=None,
        test_y=None
    )
    return result

# Define the search space for hyperparameter optimization
space = {
    "n_estimators": hp.uniformint("n_estimators", 50, 1000),
    "max_samples": hp.uniform("max_samples", 0.5, 1.0),
    "max_features": hp.uniform("max_features", 0.5, 1.0),
    "base_max_depth": hp.uniformint("base_max_depth", 1, 5),
    "base_min_samples_split": hp.uniformint("base_min_samples_split", 2, 3),
    "base_min_samples_leaf": hp.uniformint("base_min_samples_leaf", 1, 2),
    "bootstrap": hp.choice("bootstrap", [True, False]),
    "bootstrap_features": hp.choice("bootstrap_features", [True, False])
}

# Set the MLflow tracking URI and experiment name
remote_server_uri = "http://127.0.0.1:8080"  # Set to your MLflow server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("promotion_bagging")

# Start a new MLflow run for hyperparameter optimization
with mlflow.start_run():
    logging.info("Starting MLflow...")
    # Conduct the hyperparameter search using Hyperopt
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
    )

    # Fetch the details of the best run based on macro F1 score
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

    # Log the best parameters, macro F1 score (negated loss), and model
    mlflow.log_params(best)
    mlflow.log_metric("macro_f1_score", -best_run["loss"])


    # Extract the XGBoost model from the pipeline
    xgb_model = best_run["model"].named_steps['classifier']

    mlflow.sklearn.log_model(xgb_model, "model")

    # Print out the best parameters and corresponding macro F1 score
    print(f"Best parameters: {best}")
    print(f"Best eval macro F1 score: {-best_run['loss']}")

