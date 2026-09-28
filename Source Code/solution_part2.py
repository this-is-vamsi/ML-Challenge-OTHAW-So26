"""
Project Part 2: Structural Logic & Piecewise Regression
Author: [Satya Naga Vamsi Ganesh Manepalli/Exam ID: 73]
Task: Predict continuous variable "target02"
Approach: Structural Discovery via Decision Trees and Segmented OLS
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Configuration Constants
FEATURE_FILE = 'dataset_73.csv'
TARGET_FILE = 'target_73.csv'
TARGET_COL = 'target02'
SPLIT_THRESHOLDS = [0.20, 0.50, 0.70]
RANDOM_STATE = 42

def load_and_preprocess_data(feat_path, target_path):
    """
    Loads features and target datasets from CSV files.
    
    Args:
        feat_path (str): Path to the features CSV.
        target_path (str): Path to the targets CSV.
        
    Returns:
        tuple: (X as numpy array, y as numpy array)
    """
    X_df = pd.read_csv(feat_path)
    y_df = pd.read_csv(target_path)
    return X_df.values, y_df[TARGET_COL].values

def extract_key_features(X, y, top_n=4):
    """
    Identifies the most influential features using a Decision Tree Regressor.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        top_n (int): Number of top features to return.
        
    Returns:
        tuple: (index of primary splitting feature, list of explanatory feature indices)
    """
    dt = DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)
    dt.fit(X, y)
    
    # Map importance scores to feature indices
    importances = pd.Series(dt.feature_importances_)
    sorted_indices = importances.sort_values(ascending=False).index.tolist()
    
    primary_idx = sorted_indices[0]      # Main driver for piecewise splitting
    explanatory_indices = sorted_indices[1:top_n]  # Drivers for linear formulas
    
    return primary_idx, explanatory_indices

def perform_piecewise_regression(X, y, split_idx, exp_indices, thresholds):
    """
    Segments the dataset into linear regimes and derives OLS formulas for each.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        split_idx (int): Index of feature used for thresholds.
        exp_indices (list): Indices of features for the regression formulas.
        thresholds (list): Values at which to split the primary feature.
        
    Returns:
        np.ndarray: Global predictions across all segments.
    """
    # Create masks for four regions (R1 to R4)
    masks = [
        X[:, split_idx] <= thresholds[0],
        (X[:, split_idx] > thresholds[0]) & (X[:, split_idx] <= thresholds[1]),
        (X[:, split_idx] > thresholds[1]) & (X[:, split_idx] <= thresholds[2]),
        X[:, split_idx] > thresholds[2]
    ]
    
    global_preds = np.zeros(len(y))

    print("\n--- REGIONAL FORMULA ANALYSIS ---")
    for i, mask in enumerate(masks):
        X_reg, y_reg = X[mask], y[mask]
        
        # Add constant (intercept) for OLS and isolate explanatory features
        X_local = sm.add_constant(X_reg[:, exp_indices])
        model = sm.OLS(y_reg, X_local).fit()
        
        # Store predictions for global validation
        global_preds[mask] = model.predict(X_local)
        
        # Log regional performance and parameters
        cond_text = f"feat_{split_idx} <= {thresholds[i]}" if i < 3 else f"feat_{split_idx} > {thresholds[2]}"
        print(f"\n[REGION R{i+1}] Condition: {cond_text}")
        print(f"Sample Count: {len(y_reg)}")
        print(f"R-squared:    {model.rsquared:.6f}")
        print("Coefficients (const, " + ", ".join([f"feat_{idx}" for idx in exp_indices]) + "):")
        print(model.params)
        
    return global_preds

def main():
    """
    Main execution pipeline for target prediction and framework verification.
    """
    # 1. Prepare Data
    X, y = load_and_preprocess_data(FEATURE_FILE, TARGET_FILE)
    
    # 2. Identify Logic Drivers
    pri_idx, exp_indices = extract_key_features(X, y)
    print(f"Analysis identified feat_{pri_idx} as the structural root.")
    print(f"Linear components: {[f'feat_{i}' for i in exp_indices]}")

    # 3. Piecewise Modeling
    # This segments the space into regimes where OLS can find perfect coefficients
    predictions = perform_piecewise_regression(X, y, pri_idx, exp_indices, SPLIT_THRESHOLDS)

    # 4. Final Performance Validation
    final_r2 = r2_score(y, predictions)
    mae = np.mean(np.abs(y - predictions))
    
    print("\n" + "="*50)
    print(f"FINAL MODEL VERIFICATION")
    print(f"Global R2 Score: {final_r2:.8f}")
    print(f"Mean Absolute Error: {mae:.8f}")
    print("="*50)

if __name__ == "__main__":
    main()