"""
Project Part 1: Predictive Model Development
Author: [Satya Naga Vamsi Ganesh Manepalli/Exam ID: 73]
Task: Predict continuous variable "target01"
Architecture: Stacking Ensemble (LGBM, GBR, RF base models with XGBoost Meta-Learner)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. MODEL HYPERPARAMETERS

# Tuned via Bayesian Optimization (Optuna)
XGB_META_PARAMS = {
    'n_estimators': 952, 'max_depth': 5, 'learning_rate': 0.09230134948700897,
    'subsample': 0.7821187475655019, 'colsample_bytree': 0.9915671062864901,
    'gamma': 0.0015906691489470312, 'reg_alpha': 1.4484911899729072,
    'reg_lambda': 2.933296156927293, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0
}

LGBM_PARAMS = {
    'n_estimators': 793, 'max_depth': 5, 'learning_rate': 0.05260826378318587,
    'num_leaves': 56, 'subsample': 0.8579101978532994, 'colsample_bytree': 0.9946279443625201,
    'reg_alpha': 0.8799774660941787, 'reg_lambda': 0.7848078560732684,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1
}

GBR_PARAMS = {
    'n_estimators': 515, 'max_depth': 6, 'learning_rate': 0.054236840224597134,
    'subsample': 0.7537860363458649, 'min_samples_split': 13, 'min_samples_leaf': 10,
    'max_features': 0.9870837256478607, 'random_state': 42
}

RF_PARAMS = {
    'n_estimators': 564, 'max_depth': 16, 'min_samples_split': 2, 'min_samples_leaf': 2,
    'max_features': 0.8263881810707907, 'random_state': 42, 'n_jobs': -1
}

# 2. CORE FUNCTIONS

def load_and_preprocess_data(feature_path, target_path, eval_path):
    """
    Loads training and evaluation datasets, aligns columns, and splits features/target.
    
    Args:
        feature_path (str): Path to CSV containing training features.
        target_path (str): Path to CSV containing training target values.
        eval_path (str): Path to CSV containing evaluation features.
        
    Returns:
        tuple: (X, y, X_eval) cleaned and aligned dataframes.
    """
    df_feat = pd.read_csv(feature_path)
    df_target = pd.read_csv(target_path)
    df_eval = pd.read_csv(eval_path)

    # Robust target selection
    y = df_target['target01'] if 'target01' in df_target.columns else df_target.iloc[:, 0]

    # Clean non-feature columns
    X = df_feat.copy()
    for col in ['target01', 'target02', 'id']:
        if col in X.columns: X = X.drop(columns=[col])
        if col in df_eval.columns: df_eval = df_eval.drop(columns=[col])

    # Column alignment via combined dummies (if categorical data exists)
    X['is_train'] = 1
    df_eval['is_train'] = 0
    combined = pd.concat([X, df_eval], axis=0, ignore_index=True)
    cat_cols = combined.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

    X = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    X_eval = combined[combined['is_train'] == 0].drop(columns=['is_train'])

    return X, y, X_eval

def get_top_features(X, y, n=15):
    """
    Identifies the most important features using a baseline XGBoost model.
    """
    selector = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector.fit(X, y)
    importances = pd.Series(selector.feature_importances_, index=X.columns)
    return importances.nlargest(n).index.tolist()

# 3. EXECUTION PIPELINE

if __name__ == "__main__":
    print("Step 1: Loading and aligning data...")
    X, y, X_eval = load_and_preprocess_data('dataset_73.csv', 'target_73.csv', 'EVAL_73.csv')

    # Train/Val split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 2: Selecting Top 15 features via XGBoost Importance...")
    top_features = get_top_features(X_train, y_train, n=15)
    X_train_sel = X_train[top_features]
    X_val_sel = X_val[top_features]
    X_eval_sel = X_eval[top_features]

    print("Step 3: Initializing Stacking Ensemble...")
    # Base Learners
    base_models = [
        ('lgbm', LGBMRegressor(**LGBM_PARAMS)),
        ('gbr', GradientBoostingRegressor(**GBR_PARAMS)),
        ('rf', RandomForestRegressor(**RF_PARAMS))
    ]
    # Meta-Learner
    meta_learner = xgb.XGBRegressor(**XGB_META_PARAMS)

    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5, n_jobs=-1
    )

    print("Step 4: Training ensemble on raw target values...")
    stack.fit(X_train_sel, y_train)

    # Validation Check
    val_preds = stack.predict(X_val_sel)
    print("\n" + "="*40)
    print(f"VALIDATION R2 SCORE: {r2_score(y_val, val_preds):.5f}")
    print(f"VALIDATION RMSE:     {np.sqrt(mean_squared_error(y_val, val_preds)):.5f}")
    print("="*40)

    print("\nStep 5: Exporting final predictions...")
    eval_preds = stack.predict(X_eval_sel)
    output = pd.DataFrame(eval_preds, columns=['target01'])
    output.to_csv('EVAL_target01_73.csv', index=False)
    print("Process Complete. Results saved to 'EVAL_target01_73.csv'")