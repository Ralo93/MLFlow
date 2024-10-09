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

from src.features.features import create_density_features, create_structural_strength_features, \
    create_foundation_roof_features, create_geographic_risk_features, create_age_features, \
    create_family_density_feature, create_secondary_use_features, create_superstructure_variety, \
    create_position_stability_feature, create_interaction_features, get_right_skewed_columns, apply_log_transformation

# Define the search space for hyperparameter optimization
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("Loading Data...")
data = pd.read_csv(r'../data/eq_2.csv')
logging.info("Done")

numerical_df = data.select_dtypes(exclude=['object'])

# Calculate the 99th percentile of the 'age' column


# Get the right-skewed columns (excluding binary columns)
right_skewed_cols = get_right_skewed_columns(numerical_df)
numerical_df = apply_log_transformation(numerical_df, right_skewed_cols)
data[right_skewed_cols] = numerical_df[right_skewed_cols]


pct = np.percentile(data.loc[:, 'age'].fillna(np.mean(data.loc[:, 'age'])), 99)
print(pct)

# Add a new column 'old' to indicate if the age exceeds the 99th percentile
data['old'] = np.where(data['age'] >= pct, 1, 0)

# Cap the age to 100 where 'old' column is 1
data.loc[data['old'] == 1, 'age'] = 100

#from features import create_density_features


def apply_feature_engineering(df):
    df = create_density_features(df)
    df = create_structural_strength_features(df)
    df = create_foundation_roof_features(df)
    df = create_geographic_risk_features(df)
    df = create_age_features(df)
    df = create_family_density_feature(df)
    df = create_superstructure_variety(df)
    df = create_position_stability_feature(df)
    df = create_interaction_features(df)
    return df


data = apply_feature_engineering(data)


x = data.drop(columns=['damage_grade'])
y = data.damage_grade
y = y.replace({1: 0, 2: 1, 3: 2})

logging.info("Starting Preprocessing...")

numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int32', 'int64', 'float64']]

base_encoder_columns = ['land_surface_condition',
 'foundation_type',
 'roof_type',
 'ground_floor_type',
 'other_floor_type',
 'position',
 'plan_configuration',
 'legal_ownership_status']

base_encoder = Pipeline(steps=[('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))])

# Combine the transformations using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('base_name', base_encoder, base_encoder_columns),  # TargetEncoder for 'town'
    ('num', 'passthrough', numerical_cols)  # Pass numerical columns through without transformation
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


def downsample_data(train_data, class_ratios):
    """
    Downsample the majority class according to the specified ratios.

    Parameters:
    - train_data: DataFrame with the training data including target column
    - class_ratios: Dictionary specifying the desired ratio for each class

    Returns:
    - downsampled DataFrame
    """
    # Identify classes
    majority_class = train_data[train_data['damage_grade'] == 0]
    minority_class1 = train_data[train_data['damage_grade'] == 1]
    minority_class2 = train_data[train_data['damage_grade'] == 2]

    # Calculate the number of samples for each class according to the ratios
    n_minority_class1 = len(minority_class1)
    n_minority_class2 = len(minority_class2)

    # Adjust the size of the majority class according to the provided ratios
    majority_class_downsampled = resample(majority_class,
                                          replace=True,
                                          n_samples=int(class_ratios[0] * (n_minority_class1 + n_minority_class2)),
                                          random_state=42)

    minority_class1_resampled = resample(minority_class1,
                                         replace=True,
                                         n_samples=int(class_ratios[1] * (n_minority_class1 + n_minority_class2)),
                                         random_state=42)

    minority_class2_resampled = resample(minority_class2,
                                         replace=True,
                                         n_samples=int(class_ratios[2] * (n_minority_class1 + n_minority_class2)),
                                         random_state=42)

    # Combine the downsampled majority class and minority classes
    train_downsampled = pd.concat([majority_class_downsampled, minority_class1_resampled, minority_class2_resampled])

    # Shuffle the downsampled data
    return train_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)



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

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    train_accuracy_scores, train_precision_scores, train_recall_scores, train_f1_scores = [], [], [], []

    with mlflow.start_run(nested=True):
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            logging.info(f"Starting fold {fold_idx + 1}...")

            # Split the data
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            # Combine features and target into a single DataFrame for downsampling
            train_data = pd.concat([X_train_fold, y_train_fold], axis=1)

            # Specify the desired class ratios (you can adjust these values)
            class_ratios = {0: 0.33, 1: 0.33, 2: 0.33}  # Adjust the distribution ratio for each class

            # Downsample the data according to the specified ratios
            #train_downsampled = downsample_data(train_data, class_ratios)

            # Separate features and target after downsampling
            X_train_downsampled = train_data.drop('damage_grade', axis=1)
            y_train_downsampled = train_data['damage_grade']

            print('Training X_shape, y_counts:')
            print(X_train_downsampled.shape)
            print(y_train_downsampled.value_counts())

            print('Validation X_shape, y_counts:')
            print(X_valid_fold.shape)
            print(y_valid_fold.value_counts())

            # Train the model on downsampled data
            rf_pipe.fit(X_train_downsampled, y_train_downsampled)

            train_preds = rf_pipe.predict(X_train_downsampled)

            # Calculate evaluation metrics for training data (micro-average)
            train_accuracy = accuracy_score(y_train_downsampled, train_preds)
            train_precision = precision_score(y_train_downsampled, train_preds, average='micro')
            train_recall = recall_score(y_train_downsampled, train_preds, average='micro')
            train_f1 = f1_score(y_train_downsampled, train_preds, average='micro')

            # Predict on the original validation set
            valid_preds = rf_pipe.predict(X_valid_fold)

            # Calculate evaluation metrics (micro-average)
            fold_accuracy = accuracy_score(y_valid_fold, valid_preds)
            fold_precision = precision_score(y_valid_fold, valid_preds, average='micro')
            fold_recall = recall_score(y_valid_fold, valid_preds, average='micro')
            fold_f1 = f1_score(y_valid_fold, valid_preds, average='micro')

            # Append metrics to lists for training
            train_accuracy_scores.append(train_accuracy)
            train_precision_scores.append(train_precision)
            train_recall_scores.append(train_recall)
            train_f1_scores.append(train_f1)

            # Append metrics to lists for validation
            accuracy_scores.append(fold_accuracy)
            precision_scores.append(fold_precision)
            recall_scores.append(fold_recall)
            f1_scores.append(fold_f1)

        # Log training metrics with MLflow
        mlflow.log_metric("train_cv_mean_accuracy", np.mean(train_accuracy_scores))
        mlflow.log_metric("train_cv_std_accuracy", np.std(train_accuracy_scores))
        mlflow.log_metric("train_cv_mean_precision", np.mean(train_precision_scores))
        mlflow.log_metric("train_cv_std_precision", np.std(train_precision_scores))
        mlflow.log_metric("train_cv_mean_recall", np.mean(train_recall_scores))
        mlflow.log_metric("train_cv_std_recall", np.std(train_recall_scores))
        mlflow.log_metric("train_cv_mean_f1_score", np.mean(train_f1_scores))
        mlflow.log_metric("train_cv_std_f1_score", np.std(train_f1_scores))

        # Log validation metrics with MLflow
        mlflow.log_metric("cv_mean_accuracy", np.mean(accuracy_scores))
        mlflow.log_metric("cv_std_accuracy", np.std(accuracy_scores))
        mlflow.log_metric("cv_mean_precision", np.mean(precision_scores))
        mlflow.log_metric("cv_std_precision", np.std(precision_scores))
        mlflow.log_metric("cv_mean_recall", np.mean(recall_scores))
        mlflow.log_metric("cv_std_recall", np.std(recall_scores))
        mlflow.log_metric("cv_mean_f1_score", np.mean(f1_scores))
        mlflow.log_metric("cv_std_f1_score", np.std(f1_scores))

        # Log additional metrics (AUC, log loss, etc.)
        #mlflow.log_metric("cv_mean_auc", np.mean(auc_scores))
        #mlflow.log_metric("cv_std_auc", np.std(auc_scores))
        #mlflow.log_metric("cv_mean_logloss", np.mean(logloss_scores))
        #mlflow.log_metric("cv_std_logloss", np.std(logloss_scores))

        mlflow.set_tag("tag", "FE, logtransform and old column")

        logging.info("Cross-Validation completed")

        return {"loss": -np.mean(f1_scores), "status": STATUS_OK, "model": rf_pipe}


def objective(params):
    return cross_validate_with_downsampling(params, x, y)


# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/earthquake1")

# Start a new MLflow run for hyperparameter optimization
with mlflow.start_run():
    logging.info("Starting MLflow with Cross-Validation...")

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials
    )

    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    mlflow.log_params(best)

    # Change the metric logging to micro F1-score
    mlflow.log_metric("micro_f1_score", -best_run["loss"])

    xgb_model = best_run["model"].named_steps['xgboost']


    #mlflow.xgboost.log_model(xgb_model, "model", format="json")
    # Save the entire pipeline (preprocessor + model) with MLflow
    mlflow.sklearn.log_model(rf_pipe, "model")

    print(f"Best parameters: {best}")
    print(f"Best eval micro F1 score: {-best_run['loss']}")
