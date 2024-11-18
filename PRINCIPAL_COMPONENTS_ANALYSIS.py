'''Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms a large set of variables into a smaller one while retaining as much variance (information) as possible. It is widely used in exploratory data analysis, data visualization, noise reduction, and feature extraction.

In this demonstration, we will apply PCA using the scikit-learn library in Python to reduce the dimensionality of a dataset, while retaining most of its variance. We will use the Iris dataset as an example, which is a classic dataset in machine learning.

1. Import Libraries and Load Dataset
We'll use the Iris dataset, which has 4 features (sepal length, sepal width, petal length, and petal width) and 3 target classes (setosa, versicolor, virginica).'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the data to a pandas DataFrame for easier handling
X_df = pd.DataFrame(X, columns=iris.feature_names)

'''2. Standardize the Data
PCA works by finding directions (principal components) that maximize variance in the data. To ensure that features with larger scales do not dominate the analysis, we need to standardize the dataset (zero mean, unit variance).'''

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# Verify the mean and standard deviation after scaling
print("Mean after scaling: ", np.mean(X_scaled, axis=0))
print("Standard deviation after scaling: ", np.std(X_scaled, axis=0))

'3. Apply PCA
'Now, we apply PCA to reduce the dimensionality of the dataset.
# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce the data to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Check the explained variance ratio for each principal component
print("Explained variance ratio for each component: ", pca.explained_variance_ratio_)
print("Total explained variance: ", np.sum(pca.explained_variance_ratio_))

'''4. Visualize the Results
After performing PCA, we can plot the data in the new 2D space defined by the first two principal components.'''

# Step 4: Visualize the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100, alpha=0.7)
plt.colorbar(label='Species')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()

