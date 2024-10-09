import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler, MinMaxScaler
from category_encoders import BaseNEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import f1_score
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


# Set up basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("Loading Data...")
data = pd.read_csv(r'../data/train_clean2.csv')
logging.info("Done")

x = data.drop(columns=['Promoted_or_Not'])
y = data.Promoted_or_Not

logging.info("Starting Preprocessing...")

numerical_df = data.select_dtypes(exclude=['object'])
categorical_df = data.select_dtypes(include=['object'])

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in x.columns if x[cname].nunique() < 800 and x[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

numerical_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                  ('scaler', StandardScaler()) ])

#create categorical transformer
categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                            ])

base_encoder_columns = ['Division', 'Qualification', 'Channel_of_Recruitment', 'State_Of_Origin', 'Foreign_schooled', 'Marital_Status', 'Previous_IntraDepartmental_Movement', 'No_of_previous_employers', 'Gender']
base_encoder = Pipeline(steps=[
    ('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))
])
# Combine the transformations using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('base_name', base_encoder, base_encoder_columns),  # TargetEncoder for 'town'
    ('num', numerical_transformer, numerical_cols)
])

logging.info("Done.")

logging.info("Starting Resampling...")
from sklearn.utils import resample

# Combine features and target into a single DataFrame for resampling
df_train = pd.concat([X_train, y_train], axis=1)

# Identify the majority and minority classes
majority_class = df_train[df_train['Promoted_or_Not'] == 0]
minority_class = df_train[df_train['Promoted_or_Not'] == 1]

# Upsample the minority class
minority_class_upsampled = resample(
    minority_class,
    replace=True,            # Sample with replacement
    n_samples=len(majority_class),  # Match majority class
    random_state=42          # For reproducibility
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


#params = {
#    'n_estimators': 200,
#    'learning_rate': 0.1,
#    'max_depth': 6,
#    'subsample': 0.8,
#    'lambda': 1,
#    'alpha': 0,
#}

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=3,
    random_state=42,
    subsample=0.9,
    gamma=3,  # Increased from default
    reg_alpha=1,    # Added L1 regularization
    #reg_lambda=1,   # Added L2 regularization (can also be increased)
    eval_metric='mlogloss'
)

# Model pipeline
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgboost', xgb)
])

# Preprocessing of training data, fit model
#rf_pipe.fit(X_train, y_train)

# Preprocessing of training data, fit model after upsampling!
#rf_pipe.fit(X_train_upsampled, y_train_upsampled)

# Preprocessing of validation data, get predictions
#rf_preds = rf_pipe.predict(X_test)
logging.info("Done")

def train_model(params, train_x, train_y, valid_x, valid_y, test_x=None, test_y=None):
    # Enable autologging for XGBoost within the pipeline

    logging.info("Starting Training...")
    mlflow.xgboost.autolog()
    # Set the hyperparameters for the XGBoost model in the pipeline
    rf_pipe.set_params(xgboost__n_estimators=params['n_estimators'],
                       xgboost__learning_rate=params['learning_rate'],
                       xgboost__max_depth=params['max_depth'],
                       xgboost__subsample=params['subsample'],
                       xgboost__lambda=params['lambda'],
                       xgboost__colsample_bytree=params['colsample_bytree'],
                       xgboost__min_child_weight=int(params['min_child_weight']),
                       xgboost__alpha=params['alpha'])

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

        # Calculate evaluation metrics
        #eval_f1_score = f1_score(valid_y, valid_preds, average='macro')

        # Log parameters and results to MLflow (autologging handles this automatically)
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

        mlflow.set_tag("tag", "clean_train_data2")


        # Return the model and loss
        return {"loss": -eval_f1_score, "status": STATUS_OK, "model": rf_pipe}


def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        train_x=X_train_upsampled,
        train_y=y_train_upsampled,
        valid_x=X_test,
        valid_y=y_test,
        test_x=None,
        test_y=None,
    )
    return result

# Define the search space for hyperparameter optimization
space = {
    "n_estimators": hp.uniformint("n_estimators", 100, 400),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
    "max_depth": hp.uniformint("max_depth", 3, 5),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
    "subsample": hp.uniform("subsample", 0.7, 0.95),
    "min_child_weight": hp.uniformint("min_child_weight", 1, 10),
    "lambda": hp.uniform("lambda", 0, 2),
    "alpha": hp.uniform("alpha", 0, 2),
}

# Set the MLflow tracking URI and experiment name
remote_server_uri = "http://127.0.0.1:8080"  # Set to your MLflow server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("/promotion_second")

# Start a new MLflow run for hyperparameter optimization
with mlflow.start_run():
    logging.info("Starting Mlflow...")
    # Conduct the hyperparameter search using Hyperopt
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )

    # Fetch the details of the best run based on macro F1 score
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

    # Log the best parameters, macro F1 score (negated loss), and model
    mlflow.log_params(best)
    mlflow.log_metric("macro_f1_score", -best_run["loss"])

    # Extract the XGBoost model from the pipeline
    xgb_model = best_run["model"].named_steps['xgboost']

    mlflow.xgboost.log_model(xgb_model, "model")

    # Print out the best parameters and corresponding macro F1 score
    print(f"Best parameters: {best}")
    print(f"Best eval macro F1 score: {-best_run['loss']}")

