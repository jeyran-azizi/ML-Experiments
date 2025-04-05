# 🚢 Titanic Survival Prediction

This project uses the Titanic dataset to predict passenger survival based on personal and travel information.

## Dataset

- Source: Kaggle Titanic dataset
- Features: sex, age, fare, Pclass, SibSp, Parch
- Target: survived (0 = No, 1 = Yes)

## Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

## Accuracy

- Logistic Regression: ~82%
- KNN: ~78%
- Decision Tree: ~79%

## Steps

1. Load and clean the dataset (handle missing values, convert categorical variables)
2. Perform exploratory data analysis (EDA)
3. Feature selection and encoding
4. Train-test split
5. Train models and evaluate

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Run the Project

Open the notebook:

```bash
jupyter notebook Titanic~.ipynb
