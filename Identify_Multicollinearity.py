'''Multicollinearity occurs when two or more predictor variables in a regression model are highly correlated with each other. This can lead to unreliable estimates of regression coefficients, making the model's results difficult to interpret.

There are several ways to detect multicollinearity:

Correlation Matrix: Checking pairwise correlations between variables.
Variance Inflation Factor (VIF): A more sophisticated technique to quantify the severity of multicollinearity.
Condition Number: A numerical measure that indicates whether the matrix of predictors is ill-conditioned, potentially indicating multicollinearity.'''

'1. Detecting Multicollinearity Using the Correlation Matrix
'A high correlation (e.g., > 0.9 or < -0.9) between two variables indicates potential multicollinearity.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import numpy as np

# Step 1: Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Step 2: Convert to a DataFrame for easier manipulation
X_df = pd.DataFrame(X, columns=boston.feature_names)

# Step 3: Calculate the correlation matrix
correlation_matrix = X_df.corr()

# Step 4: Plot the correlation matrix using seaborn heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for Boston Housing Dataset', fontsize=16)
plt.show()

'''2. Detecting Multicollinearity Using Variance Inflation Factor (VIF)
The Variance Inflation Factor (VIF) quantifies how much the variance of a regression coefficient is inflated due to collinearity with other predictors. A high VIF (> 10) indicates potential multicollinearity.'''

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Step 1: Add a constant to the dataset to account for the intercept in the regression model
X_df_const = add_constant(X_df)

# Step 2: Calculate the VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X_df_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_df_const.values, i) for i in range(X_df_const.shape[1])]

# Step 3: Display the VIF values
print(vif_data)

'''3. Condition Number
Another technique to detect multicollinearity is by calculating the Condition Number of the matrix of predictors. A high condition number (> 30) indicates ill-conditioning and potential multicollinearity.'''

# Step 1: Calculate the condition number
from numpy.linalg import cond

# Condition number for the predictor matrix
condition_number = cond(X_df_const.values)

# Step 2: Display the condition number
print(f"Condition Number: {condition_number}")

