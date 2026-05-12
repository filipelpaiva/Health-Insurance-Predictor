import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_utils import engineer_features


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'insurance.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_FEATURE_COLS = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
TARGET_COL = 'charges'
MODEL_NAME = 'health-insurance-cost-regressor'


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'Dataset not found at {DATA_PATH}')
    return pd.read_csv(DATA_PATH)


def build_pipeline():
    numeric_features = [
        'age',
        'bmi',
        'children',
        'is_smoker',
        'is_obese',
        'age_squared',
        'bmi_age_interaction',
        'family_size',
        'smoker_bmi_interaction',
    ]
    categorical_features = ['sex', 'smoker', 'region']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_features),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipeline = Pipeline([
        ('features', FunctionTransformer(engineer_features, validate=False)),
        ('preprocessor', preprocessor),
        ('regressor', regressor),
    ])

    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model, numeric_features, categorical_features


def build_dataset_summary(df: pd.DataFrame) -> dict:
    target = df[TARGET_COL].astype(float)
    return {
        'rows': int(len(df)),
        'features': list(RAW_FEATURE_COLS),
        'target': TARGET_COL,
        'target_min': float(target.min()),
        'target_median': float(target.median()),
        'target_mean': float(target.mean()),
        'target_max': float(target.max()),
        'missing_values': int(df[RAW_FEATURE_COLS + [TARGET_COL]].isna().sum().sum()),
    }


def build_model_metadata(model, numeric_features, categorical_features, feature_names, metrics, dataset_summary):
    return {
        'model_name': MODEL_NAME,
        'model': model,
        'raw_features': RAW_FEATURE_COLS,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_names': feature_names,
        'feature_groups': {
            'raw': RAW_FEATURE_COLS,
            'engineered_numeric': numeric_features,
            'categorical': categorical_features,
        },
        'metrics': metrics,
        'dataset_summary': dataset_summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train health insurance cost model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data()
    missing = [col for col in RAW_FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    X = df[RAW_FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()
    dataset_summary = build_dataset_summary(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model, numeric_features, categorical_features = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'test_size': int(len(X_test)),
        'train_size': int(len(X_train)),
    }

    print('=== Dataset ===')
    print('Rows loaded:', dataset_summary['rows'])
    print('Target mean:', round(dataset_summary['target_mean'], 2))
    print('Target median:', round(dataset_summary['target_median'], 2))
    print('=== Split ===')
    print('Train size:', len(X_train), 'Test size:', len(X_test))
    print('=== Metrics ===')
    print('MAE:', round(mae, 2))
    print('RMSE:', round(rmse, 2))
    print('R2:', round(r2, 4))

    fitted_pipeline = model.regressor_
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out().tolist()

    model_path = os.path.join(MODEL_DIR, 'health_model.pkl')
    artifact = build_model_metadata(
        model=model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_names=feature_names,
        metrics=metrics,
        dataset_summary=dataset_summary,
    )
    joblib.dump(artifact, model_path)
    print('Model saved to', model_path)


if __name__ == '__main__':
    main()
