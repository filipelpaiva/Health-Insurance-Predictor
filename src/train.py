import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold

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
    
    df = pd.read_csv('data/insurance.csv')
    
    # Define features and target
    RAW_FEATURE_COLS = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    TARGET_COL = 'charges'

    X = df[RAW_FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Noise 
    print("Injecting Gaussian noise into training data (simulating sensor/measurement error)...")
    
    # Add a slight random variation (mean=0, standard deviation=1.0) to BMI
    noise_bmi = np.random.normal(loc=0, scale=1.0, size=len(X_train))
    X_train['bmi'] = X_train['bmi'] + noise_bmi
    
    # Add a tiny bit of noise to age (e.g., people rounding their age)
    noise_age = np.random.normal(loc=0, scale=0.5, size=len(X_train))
    X_train['age'] = X_train['age'] + noise_age
    # ==========================================

    # Define the Preprocessor
    numeric_features = ['age', 'bmi', 'children'] 
    categorical_features = ['sex', 'smoker', 'region']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_features),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    # Define Candidate Models
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42, learning_rate=0.1)
    }

    best_cv_score = -float('inf')
    best_model_name = ""
    champion_model = None

    print("\nStarting 5-Fold Model Comparison...")
    
    # Define the 5-Fold Cross Validation strategy
    from sklearn.model_selection import KFold, cross_val_score 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate Candidates
    for name, regressor in candidate_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor),
        ])

        model = TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)

        # Calculate R2 across 5 folds
        r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

        mean_r2 = r2_scores.mean()
        std_r2 = r2_scores.std()

        print(f"Model: {name:<20} | Mean R2: {mean_r2:.4f} (±{std_r2:.4f})")

        # Track the optimal model
        if mean_r2 > best_cv_score:
            best_cv_score = mean_r2
            best_model_name = name
            champion_model = model

    print(f"\nModel Comparison Complete. Optimal Model: {best_model_name}")

    # Train the optimal model on all the training data and save it
    print(f"Training {best_model_name} on full training set for production...")
    champion_model.fit(X_train, y_train)
    
    model_path = os.path.join('models', 'health_model.pkl')
    os.makedirs('models', exist_ok=True)
    joblib.dump(champion_model, model_path)
    print(f"Model successfully saved to: {model_path}")

if __name__ == '__main__':
    main()