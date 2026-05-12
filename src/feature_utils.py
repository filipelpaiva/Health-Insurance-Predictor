import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['is_smoker'] = (out['smoker'].astype(str).str.lower() == 'yes').astype(int)
    out['is_obese'] = (out['bmi'] >= 30).astype(int)
    out['age_squared'] = out['age'] ** 2
    out['bmi_age_interaction'] = out['bmi'] * out['age']
    out['family_size'] = out['children'] + 1
    out['smoker_bmi_interaction'] = out['bmi'] * out['is_smoker']
    return out
