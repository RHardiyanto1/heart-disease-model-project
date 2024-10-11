# Predicting Heart Disease Diagnosis Using Patient Medical Data

This project aims to predict the presence of heart disease using medical data collected during routine hospital visits. By analyzing factors such as age, blood pressure, cholesterol, and other health indicators, the model helps identify individuals who are at risk, enabling early intervention and potentially better treatment outcomes.

## Full Analysis

The full report can be found in the .md file or viewed through my [website](https://rhardiyanto1.github.io/posts/Heart-Disease-Project-Report/).

## Project Objective

To develop a predictive model that estimates the likelihood of heart disease in patients based on medical data, and to highlight the most significant factors contributing to the diagnosis.

## Data Sources

- **Cleveland Heart Disease Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## Methods and Tools

- **Data Cleaning and Preprocessing**: Python, Pandas
- **Exploratory Data Analysis (EDA)**: Visualizations using Matplotlib and Seaborn
- **Machine Learning Model**: Logistic Regression with regularization (L1, L2, Elastic Net) using scikit-learn and statsmodels to predict heart disease risk
- **Hyperparameter Tuning**: Used `GridSearchCV` to find the optimal model parameters

## Key Findings

- **Age, Resting Blood Pressure, and ST Depression Are Important Predictors**: Older age, higher resting blood pressure, and greater ST depression were associated with higher heart disease risk.
- **Regularization Techniques Improved Model Performance**: Applying Elastic Net regularization helped achieve a balanced model with an accuracy of **91.67%**, improving upon the initial baseline.
- **Simplifying the Target Variable Made the Model More Robust**: Changing the original target variable to a binary classification (heart disease or no heart disease) balanced the dataset and improved prediction accuracy.

## Repository Contents

- **data/**: Contains the dataset used in the project
- **images/**: Includes visualizations from the exploratory data analysis and model results
- **notebooks/**: Jupyter Notebooks detailing data cleaning, EDA, model development, and evaluation
- **powerpoint/**: Presentation slides summarizing the project's goals, methods, and key results
- **Heart Disease Project Report.md**: The main project report, outlining the entire process

## Future Research Directions

- **Try Advanced Models**: Explore more complex algorithms like Random Forests or Gradient Boosting to potentially improve accuracy.
- **Add More Data Sources**: Incorporate additional health metrics or datasets for a more comprehensive analysis.
- **Feature Engineering**: Create new features or interaction terms to capture more complex patterns in the data.

## Acknowledgements

- UCI Machine Learning Repository for providing the Cleveland Heart Disease dataset used in this project.
