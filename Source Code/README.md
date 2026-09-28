# Project Part 1: Predictive Model Development

## Project Overview
This project focuses on predicting the continuous variable `target01` using a provided dataset of 273 features. The objective was to develop a model that maximizes predictive accuracy ($R^2$) and minimizes error (RMSE).

## Technical Approach
The solution follows a systematic machine learning pipeline:
1. **Feature Selection:** Used XGBoost Importance to isolate the top 15 features, capturing non-linear interactions while reducing noise.
2. **Model Selection:** Employed a diverse set of tree-based learners (LGBM, GBR, and Random Forest) to capture various patterns in the data.
3. **Hyperparameter Optimization:** All models were tuned using Bayesian Optimization (Optuna) to ensure peak individual performance.
4. **Ensemble Architecture:** A two-level Stacking Regressor was implemented, using an optimized XGBoost as the meta-learner to blend base predictions.

## Files in this Submission
- `solution_part1.py`: The main execution script.
- `EVAL_target01_73.csv`: Final predictions for the evaluation dataset.
- `Report_73.pdf`: Detailed technical report and methodology justification.

## Requirements
To run the code, ensure the following libraries are installed:
```bash
pip install pandas numpy xgboost lightgbm scikit-learn

## References & Resources
- [Scikit-Learn Stacking Regressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) - Official guide on ensemble layering.
- [XGBoost Documentation](https://xgboost.readthedocs.io/) - Detailed technical reference for the Gradient Boosting Framework.
- [Optuna Hyperparameter Optimization GitHub](https://github.com/optuna/optuna) - The library used for Bayesian search logic.
- [Kaggle Ensemble Guide](https://www.kaggle.com/code/amirmotefaker/stacking-ensemble-learning) - A community resource on the effectiveness of stacking diverse models.

=======================================================================================


# Project Part 2: Piecewise Regression Framework

## Project Overview
Part 2 of the project focuses on predicting the continuous variable `target02`. Unlike `target01`, which was best handled by complex ensembles, `target02` exhibits a structural, regime-based logic. This framework identifies the splitting thresholds of the data to create localized linear models, resulting in an extremely high-precision fit.

## Technical Approach
The solution employs a **Hybrid Decision Tree-OLS** architecture:

1. **Root Driver Identification:** A Decision Tree Regressor is used to scan all 273 features to find the "Structural Root"—the single feature that accounts for the most significant variance shifts (structural breaks).
2. **Data Segmentation (Piecewise Logic):** Based on the identified root feature, the data is segmented into four distinct regimes using optimized thresholds ($0.2, 0.5, 0.7$).
3. **Regional OLS Modeling:** Within each segment, an Ordinary Least Squares (OLS) regression is performed. This identifies the specific linear formula (coefficients and intercept) that governs that particular regime.
4. **Global Synthesis:** Regional predictions are unified into a global prediction vector for final validation.



## Files in this Submission
- `solution_part2.py`: The main execution script utilizing `statsmodels` for regional analysis.
- `framework_73.py`: The final production framework that generates predictions using the identified piecewise logic.
- `Report_73.pdf`: Technical justification of the piecewise linear approach.

## Requirements
To run the code, ensure the following libraries are installed:
```bash
pip install pandas numpy scikit-learn statsmodels

## References & Resources
- [Statsmodels OLS Documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) - Technical reference for Ordinary Least Squares analysis.
- [Piecewise Regression Theory](https://github.com/pwc-data-science/piecewise-regression) - Concepts behind segmenting data into different linear regimes.
- [Decision Trees for Structural Discovery](https://scikit-learn.org/stable/modules/tree.html) - How tree-based splits define logical breakpoints in datasets.