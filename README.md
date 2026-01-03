# Wine Quality Classification: Lasso vs Random Forest

This repository contains the code and analysis for a Machine Learning assignment focused on **binary classification and model comparison** using the Wine Quality dataset.

The objective is to frame a real-world prediction problem, apply two contrasting modelling approaches, and evaluate their performance with an emphasis on **interpretability, validation, and decision-making trade-offs** rather than accuracy alone.

## Problem Definition
Wine quality is originally measured on an ordinal scale. For this study, the problem is reframed as a **binary classification task**:

- **Good wine:** quality ≥ 6  
- **Bad wine:** quality < 6  

This framing reflects a realistic decision boundary often used in applied settings (e.g. accept vs reject, promote vs filter).

## Data
- Source: UCI Wine Quality dataset  
- Data used:
  - Red wine dataset
  - White wine dataset
- Both datasets are merged and a categorical variable (`type`) is introduced to distinguish wine type.

No missing values are present in the data.

## Data Preparation
Key preprocessing steps include:
- Merging red and white wine datasets
- Converting wine type to a factor
- Creating a binary response variable (`quality_bin`)
- Stratified train–test split (75% train / 25% test) using `caret`
- Exploratory Data Analysis (EDA) to understand:
  - Feature distributions
  - Differences between red and white wines
  - Relationships between predictors

## Modelling Approach
Two models are implemented, selected from distinct methodological families:

### 1. Lasso Logistic Regression
- Implemented using `glmnet` via the `caret` framework
- 10-fold cross-validation for tuning
- Regularisation parameter (`lambda`) selected based on classification accuracy
- Motivation:
  - Feature selection
  - Interpretability
  - Bias–variance control

### 2. Random Forest Classifier
- Implemented using `randomForest`
- Default ensemble configuration
- Variable importance extracted to understand key drivers of prediction
- Additional analysis includes:
  - Proximity-based MDS visualisation
  - Bootstrap-based RMSE distribution for robustness

## Model Evaluation
Models are evaluated on a held-out test set using:
- Accuracy
- Kappa
- Sensitivity and Specificity
- Precision
- Balanced Accuracy

Confusion matrices are used to support interpretation beyond a single metric.

## Model Comparison
The comparison highlights clear trade-offs:

- **Lasso** provides:
  - Simpler decision boundaries
  - Greater interpretability
  - Explicit feature shrinkage

- **Random Forest** provides:
  - Higher flexibility
  - Stronger performance on non-linear relationships
  - Richer insight into feature importance

Rather than selecting a single “best” model, the analysis focuses on **contextual suitability and decision impact**.

## Tools & Libraries
- **Language:** R
- **Core packages:**
  - caret
  - glmnet
  - randomForest
  - ggplot2
  - dplyr
  - tidyr
  - GGally

## Repository Structure

