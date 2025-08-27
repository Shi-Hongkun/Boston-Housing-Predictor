# Boston Housing Predictor

A project that analyzes temporal drift effects in housing price prediction by comparing chronological splits (first 70% older houses vs last 30% newer houses) with random split baselines.

## Project Overview

This project addresses the challenge of model performance degradation when training on historical data and testing on newer data. It implements:

- **Chronological Split**: Train on first 70% of data (older houses), test on last 30% (newer houses)
- **Random Split Baseline**: Standard 70/30 random split for comparison
- **AGE-based Split**: Alternative temporal split based on house age
- **Multiple Models**: Ridge regression and Gradient Boosting with comprehensive evaluation
- **Drift Analysis**: Statistical analysis of feature distribution shifts between train/test sets

## Dataset

The dataset (`data/housing.csv`) contains 506 samples with 13 features:
- **Features**: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
- **Target**: MEDV (median house value in $1000s)

## Quick Start

### 1. Install uv

### 2. Setup Project
```bash
uv sync
```

### 3. Start Analysis
run in your IDE, or

```bash
uv run jupyter lab
```
Open `ref/the-boston-housing-dataset.ipynb` in your browser.

## Project Structure

```
Boston-Housing-Predictor/
├── data/
│   └── housing.csv          # Boston Housing dataset
├── ref/
│   └── the-boston-housing-dataset.ipynb  # A copy of the Kaggle's script, for ref only
├── boston-housing-predictor.ipynb # Complete analysis notebook
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # Locked dependency versions
└── README.md               # This file
```

## Analysis Notebook

The notebook `boston-housing-predictor.ipynb` contains a complete implementation:

- **7 main sections** covering the complete analysis workflow
- **Three split strategies**: chronological, AGE-based, and random baseline
- **Two dataset versions**: keep-all vs remove-censored data
- **Two models**: Ridge regression and GradientBoosting
- **Comprehensive evaluation**: RMSE, MAE, R² across all combinations
- **Drift analysis**: Feature distribution shifts and statistical significance
- **Model explainability**: Coefficients and feature importance analysis

## Key Features

- **Reproducible Environment**: Locked dependencies with uv.lock
- **Temporal Drift Analysis**: Compare chronological vs random splits
- **Multiple Model Evaluation**: Ridge regression and Gradient Boosting
- **Statistical Analysis**: Feature distribution shifts and model explainability
- **Robustness Recommendations**: Strategies for handling temporal changes

## Dependencies

Core packages:
- `numpy`, `pandas`, `scikit-learn` - Data processing and ML
- `seaborn`, `matplotlib` - Visualization
- `scipy` - Statistical analysis
- `jupyter`, `ipykernel` - Notebook environment

## License

This project is for educational and case study purposes.
