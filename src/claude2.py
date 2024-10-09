import mlflow
import numpy as np
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold

# Define the search space for hyperparameter optimization
space = {
    "n_estimators": hp.quniform("n_estimators", 100, 1000, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
    "subsample": hp.uniform("subsample", 0.7, 1),
    "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
    "lambda": hp.loguniform("lambda", np.log(1e-8), np.log(10)),
    "alpha": hp.loguniform("alpha", np.log(1e-8), np.log(10)),
    "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(10)),
    "scale_pos_weight": hp.loguniform("scale_pos_weight", np.log(1), np.log(100))
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data():
    logging.info("Loading Data...")
    data = pd.read_csv(r'../data/train_clean2.csv')
    logging.info("Data loaded successfully.")
    return data


def create_preprocessor(X):
    logging.info("Creating preprocessor...")

    # Identify column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])

    logging.info(
        f"Preprocessor created with {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
    return preprocessor


def preprocess_data(data):
    logging.info("Starting Preprocessing...")
    X = data.drop(columns=['Promoted_or_Not'])
    y = data['Promoted_or_Not']
    preprocessor = create_preprocessor(X)
    logging.info("Preprocessing completed.")
    return preprocessor, X, y


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
        scale_pos_weight=params['scale_pos_weight'],
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )


def cross_validate(params, X, y, preprocessor):
    mlflow.xgboost.autolog()

    logging.info("Starting Cross-Validation...")

    xgb = create_model(params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('xgboost', xgb)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'logloss']}

    with mlflow.start_run(nested=True):
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            logging.info(f"Starting fold {fold_idx + 1}...")

            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            pipeline.fit(X_train_fold, y_train_fold)

            y_pred = pipeline.predict(X_valid_fold)
            y_pred_proba = pipeline.predict_proba(X_valid_fold)

            fold_metrics = {
                'accuracy': accuracy_score(y_valid_fold, y_pred),
                'precision': precision_score(y_valid_fold, y_pred, average='macro'),
                'recall': recall_score(y_valid_fold, y_pred, average='macro'),
                'f1': f1_score(y_valid_fold, y_pred, average='macro'),
                'auc': roc_auc_score(y_valid_fold, y_pred_proba, multi_class='ovr', average='macro'),
                'logloss': log_loss(y_valid_fold, y_pred_proba)
            }

            for metric, value in fold_metrics.items():
                metrics[metric].append(value)
                mlflow.log_metric(f"fold_{fold_idx + 1}_{metric}", value)

        # Log mean and std of metrics
        for metric in metrics.keys():
            mlflow.log_metric(f"cv_mean_{metric}", np.mean(metrics[metric]))
            mlflow.log_metric(f"cv_std_{metric}", np.std(metrics[metric]))

        mlflow.set_tag("model_type", "XGBoost with class weight balancing")
        mlflow.set_tag("cv_strategy", "StratifiedKFold")

        logging.info("Cross-Validation completed")

        return {"loss": -np.mean(metrics['f1']), "status": STATUS_OK, "model": pipeline}


def objective(params):
    return cross_validate(params, X, y, preprocessor)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("/promotion_cv_balanced_flexible")

    data = load_data()
    preprocessor, X, y = preprocess_data(data)

    with mlflow.start_run():
        logging.info("Starting Hyperparameter Optimization...")
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )

        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
        mlflow.log_params(best)
        mlflow.log_metric("best_macro_f1_score", -best_run["loss"])

        xgb_model = best_run["model"].named_steps['xgboost']
        mlflow.xgboost.log_model(xgb_model, "best_model")

        logging.info(f"Best parameters: {best}")
        logging.info(f"Best eval macro F1 score: {-best_run['loss']}")