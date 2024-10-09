from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import mlflow
import numpy as np
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import BaseNEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split

# Define the search space for hyperparameter optimization
space = {
    "n_estimators": hp.uniformint("n_estimators", 100, 1000),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
    "max_depth": hp.uniformint("max_depth", 3, 7),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
    "subsample": hp.uniform("subsample", 0.7, 0.95),
    "min_child_weight": hp.uniformint("min_child_weight", 1, 10),
    "lambda": hp.loguniform("lambda", np.log(1e-8), np.log(10)),
        "alpha": hp.loguniform("alpha", np.log(1e-8), np.log(10)),
        "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(10))
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("Loading Data...")
data = pd.read_csv(r'../data/train_clean2.csv')
logging.info("Done")

x = data.drop(columns=['Promoted_or_Not'])
y = data.Promoted_or_Not

logging.info("Starting Preprocessing...")

numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]
base_encoder_columns = ['Division', 'Qualification', 'Channel_of_Recruitment', 'State_Of_Origin',
                        'Foreign_schooled', 'Marital_Status', 'Previous_IntraDepartmental_Movement',
                        'No_of_previous_employers', 'Gender']

base_encoder = Pipeline(steps=[('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))])

# Combine the transformations using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('base_name', base_encoder, base_encoder_columns),
    ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_cols)
])

# Consider doing PCA!

def create_model(params):
    return XGBClassifier(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        colsample_bytree=params['colsample_bytree'],
        subsample=params['subsample'],
        min_child_weight=int(params['min_child_weight']),
        lambda_=params['lambda'],
        alpha=params['alpha'],
        gamma=params['gamma'],
        random_state=42,
        eval_metric='mlogloss'
    )

# Model pipeline
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=3,
    random_state=42,
    subsample=0.9,
    gamma=3,  # Increased from default
    reg_alpha=1,  # L1 regularization
    eval_metric='mlogloss'
)

rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('xgboost', xgb)])


# Upsample only the training data within each fold
def cross_validate_with_downsampling(params, X, y):
    mlflow.xgboost.autolog()

    logging.info("Starting Cross-Validation with Upsampling...")


    rf_pipe.set_params(
        xgboost__n_estimators=params['n_estimators'],
        xgboost__learning_rate=params['learning_rate'],
        xgboost__max_depth=params['max_depth'],
        xgboost__subsample=params['subsample'],
        xgboost__lambda=params['lambda'],
        xgboost__colsample_bytree=params['colsample_bytree'],
        xgboost__min_child_weight=int(params['min_child_weight']),
        xgboost__alpha=params['alpha']
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    train_accuracy_scores, train_precision_scores, train_recall_scores, train_f1_scores = [], [], [], []

    with mlflow.start_run(nested=True):
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            logging.info(f"Starting fold {fold_idx + 1}...")

            # Split the data
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            # Combine features and target into a single DataFrame for upsampling
            train_data = pd.concat([X_train_fold, y_train_fold], axis=1)

            # Identify majority and minority classes
            majority_class = train_data[train_data['Promoted_or_Not'] == 0]
            minority_class = train_data[train_data['Promoted_or_Not'] == 1]

            # Downsample the majority class
            majority_downsampled = resample(majority_class,
                                            replace=False,  # No replacement, to downsample
                                            n_samples=len(minority_class)*4,  # Match the number of minority class samples
                                            random_state=42)

            # Combine downsampled majority class with minority class
            train_downsampled = pd.concat([majority_downsampled, minority_class])

            # Shuffle the downsampled training data
            train_downsampled = train_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

            # Separate features and target after downsampling
            X_train_downsampled = train_downsampled.drop('Promoted_or_Not', axis=1)
            y_train_downsampled = train_downsampled['Promoted_or_Not']

            print('Training X_shape, y_counts:')
            print(X_train_downsampled.shape)
            print(y_train_downsampled.value_counts())

            print('Validation X_shape, y_counts:')
            print(X_valid_fold.shape)
            print(y_valid_fold.value_counts())

            # Train the model on upsampled data
            rf_pipe.fit(X_train_downsampled, y_train_downsampled)

            train_preds = rf_pipe.predict(X_train_downsampled)

            # Calculate evaluation metrics for training data
            train_accuracy = accuracy_score(y_train_downsampled, train_preds)
            train_precision = precision_score(y_train_downsampled, train_preds, average='macro')
            train_recall = recall_score(y_train_downsampled, train_preds, average='macro')
            train_f1 = f1_score(y_train_downsampled, train_preds, average='macro')

            # Predict on the original validation set
            valid_preds = rf_pipe.predict(X_valid_fold)

            # Calculate evaluation metrics
            fold_accuracy = accuracy_score(y_valid_fold, valid_preds)
            fold_precision = precision_score(y_valid_fold, valid_preds, average='macro')
            fold_recall = recall_score(y_valid_fold, valid_preds, average='macro')
            fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')

            # Append metrics to lists for training as well
            train_accuracy_scores.append(train_accuracy)
            train_precision_scores.append(train_precision)
            train_recall_scores.append(train_recall)
            train_f1_scores.append(train_f1)

            # Append metrics to lists
            accuracy_scores.append(fold_accuracy)
            precision_scores.append(fold_precision)
            recall_scores.append(fold_recall)
            f1_scores.append(fold_f1)

        mlflow.log_metric("train_cv_mean_accuracy", np.mean(train_accuracy_scores))
        mlflow.log_metric("train_cv_std_accuracy", np.std(train_accuracy_scores))
        mlflow.log_metric("train_cv_mean_precision", np.mean(train_precision_scores))
        mlflow.log_metric("train_cv_std_precision", np.std(train_precision_scores))
        mlflow.log_metric("train_cv_mean_recall", np.mean(train_recall_scores))
        mlflow.log_metric("train_cv_std_recall", np.std(train_recall_scores))
        mlflow.log_metric("train_cv_mean_f1_score", np.mean(train_f1_scores))
        mlflow.log_metric("train_cv_std_f1_score", np.std(train_f1_scores))

        # Log cross-validation metrics
        mlflow.log_metric("cv_mean_accuracy", np.mean(accuracy_scores))
        mlflow.log_metric("cv_std_accuracy", np.std(accuracy_scores))
        mlflow.log_metric("cv_mean_precision", np.mean(precision_scores))
        mlflow.log_metric("cv_std_precision", np.std(precision_scores))
        mlflow.log_metric("cv_mean_recall", np.mean(recall_scores))
        mlflow.log_metric("cv_std_recall", np.std(recall_scores))
        mlflow.log_metric("cv_mean_f1_score", np.mean(f1_scores))
        mlflow.log_metric("cv_std_f1_score", np.std(f1_scores))
        mlflow.log_metric("cv_mean_auc", np.std(f1_scores))
        mlflow.log_metric("cv_std_auc", np.std(f1_scores))
        mlflow.log_metric("cv_mean_auc", np.std(f1_scores))
        mlflow.log_metric("cv_std_auc", np.std(f1_scores))
        mlflow.log_metric("cv_mean_logloss", np.std(f1_scores))
        mlflow.log_metric("cv_std_logloss", np.std(f1_scores))

        mlflow.set_tag("tag", "CV with DS*4 test metrics")

        logging.info("Cross-Validation completed")

        return {"loss": -np.mean(f1_scores), "status": STATUS_OK, "model": rf_pipe}


def objective(params):
    return cross_validate_with_downsampling(params, x, y)


# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/promotion_test2")

# Start a new MLflow run for hyperparameter optimization
with mlflow.start_run():
    logging.info("Starting MLflow with Cross-Validation...")

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=3,
        trials=trials
    )

    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    mlflow.log_params(best)
    mlflow.log_metric("macro_f1_score", -best_run["loss"])

    xgb_model = best_run["model"].named_steps['xgboost']
    mlflow.xgboost.log_model(xgb_model, "model", format="json")

    print(f"Best parameters: {best}")
    print(f"Best eval macro F1 score: {-best_run['loss']}")
