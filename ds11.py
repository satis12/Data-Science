import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'path/to/your/dataset.csv' with your actual file path)
# For demonstration, let's use the Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Assuming 'X' contains features for PCA
X = df_iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Evaluate explained variance to select the appropriate number of principal components
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Visualize explained variance
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Choose the appropriate number of principal components based on the plot or a threshold
# For example, let's say you choose the first two principal components
num_components = 2

# Use the selected number of principal components
pca_selected = PCA(n_components=num_components)
X_pca_selected = pca_selected.fit_transform(X_scaled)

# Visualize the data in the reduced-dimensional space
plt.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], c=iris.target, cmap='viridis')
plt.title('PCA - Reduced-dimensional Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
