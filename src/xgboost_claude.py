import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import BaseNEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Loading Data...")
    data = pd.read_csv(r'../data/train_clean2.csv')
    logging.info("Data loaded")
    return data.drop(columns=['Promoted_or_Not']), data.Promoted_or_Not

def create_preprocessor(x):
    numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]
    base_encoder_columns = ['Division', 'Qualification', 'Channel_of_Recruitment', 'State_Of_Origin',
                            'Foreign_schooled', 'Marital_Status', 'Previous_IntraDepartmental_Movement',
                            'No_of_previous_employers', 'Gender']

    base_encoder = Pipeline(steps=[('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))])

    return ColumnTransformer(transformers=[
        ('base_name', base_encoder, base_encoder_columns),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_cols)
    ])

def create_model(params):
    return XGBClassifier(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        colsample_bytree=params['colsample_bytree'],
        subsample=params['subsample'],
        min_child_weight=int(params['min_child_weight']),
        reg_lambda=params['reg_lambda'],
        alpha=params['alpha'],
        gamma=params['gamma'],
        random_state=42,
        eval_metric='mlogloss'
    )

def cross_validate_with_downsampling(params, X, y):
    #mlflow.xgboost.autolog()
    logging.info("Starting Cross-Validation with Downsampling...")

    preprocessor = create_preprocessor(X)
    xgb = create_model(params)
    rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('xgboost', xgb)])

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = {metric: {'train': [], 'valid': []} for metric in ['accuracy', 'precision', 'recall', 'f1']}

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        logging.info(f"Starting fold {fold_idx + 1}...")

        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = pd.concat([X_train_fold, y_train_fold], axis=1)
        majority_class = train_data[train_data['Promoted_or_Not'] == 0]
        minority_class = train_data[train_data['Promoted_or_Not'] == 1]

        majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class)*4, random_state=42)
        train_downsampled = pd.concat([majority_downsampled, minority_class]).sample(frac=1, random_state=42).reset_index(drop=True)

        X_train_downsampled = train_downsampled.drop('Promoted_or_Not', axis=1)
        y_train_downsampled = train_downsampled['Promoted_or_Not']

        rf_pipe.fit(X_train_downsampled, y_train_downsampled)

        for dataset, X_data, y_data in [('train', X_train_downsampled, y_train_downsampled),
                                        ('valid', X_valid_fold, y_valid_fold)]:
            preds = rf_pipe.predict(X_data)
            for metric, func in [('accuracy', accuracy_score), ('precision', precision_score),
                                 ('recall', recall_score), ('f1', f1_score)]:
                score = func(y_data, preds, average='macro') if metric != 'accuracy' else func(y_data, preds)
                scores[metric][dataset].append(score)

    for metric in scores:
        for dataset in ['train', 'valid']:
            mlflow.log_metric(f"{dataset}_cv_mean_{metric}", np.mean(scores[metric][dataset]))
            mlflow.log_metric(f"{dataset}_cv_std_{metric}", np.std(scores[metric][dataset]))

    mlflow.set_tag("tag", "CV(4) with DS*4 claude")
    logging.info("Cross-Validation completed")

    return {"loss": -np.mean(scores['f1']['valid']), "status": STATUS_OK, "model": rf_pipe}

def objective(params):
    return cross_validate_with_downsampling(params, x, y)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("/promotion_claude")

    # Define the search space for hyperparameter optimization
    space = {
        "n_estimators": hp.uniformint("n_estimators", 100, 1000),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.uniformint("max_depth", 3, 7),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
        "subsample": hp.uniform("subsample", 0.7, 0.95),
        "min_child_weight": hp.uniformint("min_child_weight", 1, 10),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10)),
        "alpha": hp.loguniform("alpha", np.log(1e-8), np.log(10)),
        "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(10))
    }

    x, y = load_data()

    with mlflow.start_run():
        logging.info("Starting MLflow with Cross-Validation...")

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
        mlflow.log_params(best)
        mlflow.log_metric("macro_f1_score", -best_run["loss"])

        xgb_model = best_run["model"].named_steps['xgboost']
        mlflow.xgboost.log_model(xgb_model, "model", format="json")

        print(f"Best parameters: {best}")
        print(f"Best eval macro F1 score: {-best_run['loss']}")