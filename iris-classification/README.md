# 🌸 Iris Flower Classification

This project uses the famous Iris dataset to classify flowers into three species — Setosa, Versicolor, and Virginica — based on their sepal and petal dimensions.

## Dataset

- Source: scikit-learn built-in datasets
- Features: sepal length, sepal width, petal length, petal width
- Target: species (Setosa, Versicolor, Virginica)

## Models Used

- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

## Accuracy

- KNN: ~97%
- Decision Tree: ~96%

## Steps

1. Load and explore the dataset
2. Visualize with seaborn (pairplots, scatter plots)
3. Split data into train and test sets
4. Train classifiers and make predictions
5. Evaluate using accuracy score and confusion matrix

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Run the Project

Open the notebook:

```bash
jupyter notebook iris_classification.ipynb


























# Machine Learning Projects Collection

This repository contains a collection of introductory machine learning projects implemented in Python using popular libraries like Pandas, Scikit-learn, and Matplotlib. These projects demonstrate foundational concepts in data preprocessing, exploratory data analysis (EDA), classification models, and evaluation techniques.

## Projects Included

### 🌸 Iris Flower Classification
- **Dataset:** [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- **Objective:** Classify iris flowers into three species based on features like petal and sepal dimensions.
- **Key Concepts:** 
  - Data visualization
  - Label encoding
  - Train-test split
  - Model training using Decision Tree, Logistic Regression, KNN, and Random Forest
  - Evaluation with accuracy and classification report
- **Accuracy:**  
  - KNN: ~0.97  
  - Decision Tree: ~0.96

### 🚢 Titanic Survival Prediction
- **Dataset:** [Titanic dataset](https://www.kaggle.com/c/titanic)
- **Objective:** Predict whether a passenger survived based on features such as age, sex, and passenger class.
- **Key Concepts:**
  - Handling missing values
  - One-hot encoding
  - Model comparison
  - Feature importance
- **Accuracy:**  
  - Logistic Regression: ~0.82 
  - Decision Tree: ~0.79 
  - KNN: ~0.78

### 📚 Student Performance Analysis
- **Dataset:** [Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **Objective:** Predict whether a student passes based on factors such as study time, absences, and parental education.
- **Key Concepts:**
  - Categorical to numerical conversion
  - Data cleaning
  - Model performance metrics
  - Visual insights
- **Accuracy:**  
  - Logistic Regression: ~0.83
  - Decision Tree: ~0.82
  - KNN: ~0.79

## Requirements

To run these notebooks, install the required libraries using:

```bash
pip install pandas matplotlib seaborn scikit-learn

