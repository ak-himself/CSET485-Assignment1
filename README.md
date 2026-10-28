# CSET485 – AI and Society | Assignment 1

**Student:** Ankit Kumar  
**Roll Number:** S24CSEU0910

## Overview

This assignment investigates how predictive models can behave in high-stakes decisions. It uses the COMPAS recidivism dataset to compare model outputs and examine subgroup effects, omitted-variable bias, confident errors, and fairness trade-offs.

## Workflow

1. **Load and inspect data** – download the COMPAS dataset and inspect its columns, missing values, and distributions.
2. **Preprocess** – validate the target columns, remove duplicate rows, impute missing feature values, encode categorical variables, and scale numeric features.
3. **Split data** – use a 70/30 train-test split with `random_state=910`.
4. **Train models** – use linear regression to predict `decile_score` and logistic regression to predict `two_year_recid`.
5. **Compare classifications** – threshold linear predictions at their test-set median and logistic probabilities at 0.5, then measure their disagreements.
6. **Run stress tests** – compare subgroup coefficients, test omitted-variable bias, identify confident incorrect predictions, and calculate fairness metrics across decision thresholds.
7. **Review results** – use the notebook's computed tables and chart to support the written analysis and reflection.

## Tools

- Python
- Jupyter Notebook
- pandas and NumPy
- scikit-learn and statsmodels
- Matplotlib and Seaborn

The notebook is designed to run on CPU and does not require CUDA or a GPU.

## Files

- `CSET485_Assignment1_s24cseu0910.ipynb` – notebook containing the code and computations.
- `Assignment1_Report.pdf` – exported assignment report, if included in the repository.

## Reproducibility

Run the notebook cells from top to bottom. The notebook downloads the dataset and computes the model comparisons, stress-test tables, and fairness chart. Results may depend on library versions and dataset availability.
