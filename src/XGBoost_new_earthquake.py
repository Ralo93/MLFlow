from hyperopt import STATUS_OK, hp, Trials, fmin, tpe

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import resample
import numpy as np
import pandas as pd
import logging
from category_encoders import BaseNEncoder
from xgboost import XGBClassifier
import mlflow

from src.features.features import get_right_skewed_columns, create_density_features, \
    create_structural_strength_features, create_foundation_roof_features, create_geographic_risk_features, \
    create_age_features, create_position_stability_feature, create_interaction_features, create_family_density_feature, \
    create_superstructure_variety


# Custom Transformer for Log Transformation
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            # Apply log transformation (log1p to handle zeros)
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy


# Custom Transformer for Feature Engineering
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply all feature engineering functions
        #X = create_density_features(X)
        #X = create_structural_strength_features(X)
        #X = create_foundation_roof_features(X)
        #X = create_geographic_risk_features(X)
        #X = create_age_features(X)
        #X = create_family_density_feature(X)
        #X = create_superstructure_variety(X)
        #X = create_position_stability_feature(X)
        #X = create_interaction_features(X)
        return X


# Function to create preprocessing pipeline
def create_preprocessor(numerical_cols, base_encoder_columns, right_skewed_cols):
    # BaseNEncoder for categorical columns
    base_encoder = Pipeline(steps=[('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))])

    # ColumnTransformer to apply different transformers to different subsets of features
    preprocessor = ColumnTransformer(transformers=[
        ('base_name', base_encoder, base_encoder_columns),  # TargetEncoder for categorical columns
        ('log_transform', LogTransformer(columns=right_skewed_cols), right_skewed_cols),
        # Log transformation for skewed columns
        ('num', 'passthrough', numerical_cols)  # Pass through numerical columns without transformation
    ])

    return preprocessor


# Create the full pipeline
def create_full_pipeline(numerical_cols, base_encoder_columns, right_skewed_cols, model):
    # Create preprocessor pipeline
    preprocessor = create_preprocessor(numerical_cols, base_encoder_columns, right_skewed_cols)

    # Full pipeline: preprocessor + feature engineering + model
    full_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('model', model)
    ])

    return full_pipeline


# Function to run downsampling
def downsample_data(train_data, class_ratios):
    majority_class = train_data[train_data['damage_grade'] == 0]
    minority_class1 = train_data[train_data['damage_grade'] == 1]
    minority_class2 = train_data[train_data['damage_grade'] == 2]

    n_minority_class1 = len(minority_class1)
    n_minority_class2 = len(minority_class2)

    majority_class_downsampled = resample(
        majority_class, replace=True,
        n_samples=int(class_ratios[0] * (n_minority_class1 + n_minority_class2)),
        random_state=42
    )

    minority_class1_resampled = resample(
        minority_class1, replace=True,
        n_samples=int(class_ratios[1] * (n_minority_class1 + n_minority_class2)),
        random_state=42
    )

    minority_class2_resampled = resample(
        minority_class2, replace=True,
        n_samples=int(class_ratios[2] * (n_minority_class1 + n_minority_class2)),
        random_state=42
    )

    train_downsampled = pd.concat([majority_class_downsampled, minority_class1_resampled, minority_class2_resampled])

    return train_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)


# Hyperparameter optimization with cross-validation
def cross_validate_with_downsampling(params, X, y, full_pipeline):
    mlflow.xgboost.autolog()
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    accuracy_scores_train, precision_scores_train, recall_scores_train, f1_scores_train = [], [], [], []

    with mlflow.start_run(nested=True):
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            logging.info(f"Starting fold {fold_idx + 1}...")

            # Split data into training and validation sets
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            # Combine features and target for downsampling
            train_data = pd.concat([X_train_fold, y_train_fold], axis=1)
            class_ratios = {0: 0.33, 1: 0.33, 2: 0.33}

            # Downsample the training data
            #train_downsampled = downsample_data(X_train_fold, class_ratios)
            #X_train_downsampled = train_downsampled.drop('damage_grade', axis=1)
            #y_train_downsampled = train_downsampled['damage_grade']

            # Train the full pipeline
            full_pipeline.fit(X_train_fold, y_train_fold)

            train_preds = full_pipeline.predict(X_train_fold)
            valid_preds = full_pipeline.predict(X_valid_fold)

            # Calculate and store metrics
            accuracy_scores.append(accuracy_score(y_valid_fold, valid_preds))
            precision_scores.append(precision_score(y_valid_fold, valid_preds, average='micro'))
            recall_scores.append(recall_score(y_valid_fold, valid_preds, average='micro'))
            f1_scores.append(f1_score(y_valid_fold, valid_preds, average='micro'))

            # Calculate and store metrics
            accuracy_scores_train.append(accuracy_score(y_train_fold, train_preds))
            precision_scores_train.append(precision_score(y_train_fold, train_preds, average='micro'))
            recall_scores_train.append(recall_score(y_train_fold, train_preds, average='micro'))
            f1_scores_train.append(f1_score(y_train_fold, train_preds, average='micro'))

        mlflow.log_metric("cv_mean_accuracy", np.mean(accuracy_scores))
        mlflow.log_metric("cv_mean_f1_score", np.mean(f1_scores))
        mlflow.log_metric("cv_mean_accuracy", np.mean(accuracy_scores_train))
        mlflow.log_metric("cv_mean_f1_score", np.mean(f1_scores_train))

        return {"loss": -np.mean(f1_scores), "status": STATUS_OK, "model": full_pipeline}


# Run hyperparameter optimization
def objective(params):
    return cross_validate_with_downsampling(params, x, y, full_pipeline)


# Main function to set up and run everything
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load data
    logging.info("Loading Data...")
    data = pd.read_csv(r'../data/eq_2.csv')

    # Preprocess target variable and define X and y
    x = data.drop(columns=['damage_grade'])
    y = data['damage_grade'].replace({1: 0, 2: 1, 3: 2})

    # Identify numerical and categorical columns
    numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int32', 'int64', 'float64']]
    base_encoder_columns = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']

    # Identify right-skewed columns
    right_skewed_cols = get_right_skewed_columns(x.select_dtypes(exclude=['object']))

    # Create XGBoost model
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.2, max_depth=3, random_state=42, eval_metric='mlogloss')

    # Create full pipeline
    full_pipeline = create_full_pipeline(numerical_cols, base_encoder_columns, right_skewed_cols, xgb)

    # Define search space for hyperparameter tuning
    # Define search space for hyperparameter tuning
    space = {
        "n_estimators": hp.uniformint("n_estimators", 300, 800),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.5)),
        "max_depth": hp.uniformint("max_depth", 3, 6),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1),
        "subsample": hp.uniform("subsample", 0.6, 0.95),
        "min_child_weight": hp.uniformint("min_child_weight", 4, 6),
        "lambda": hp.loguniform("lambda", np.log(1e-8), np.log(10)),
        "alpha": hp.loguniform("alpha", np.log(1e-8), np.log(10)),
        "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(10))
    }

    # Hyperparameter optimization
    mlflow.set_experiment("/earthquake1")
    with mlflow.start_run():
        logging.info("Starting Hyperparameter Optimization...")
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)

        # Log best parameters and save the model
        mlflow.log_params(best)
        mlflow.sklearn.log_model(full_pipeline, "model")

        logging.info(f"Best parameters: {best}")