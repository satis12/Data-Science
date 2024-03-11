import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target_names[iris.target]

# Assume we want to predict petal length ('petal length (cm)') based on sepal length ('sepal length (cm)')
X_iris = df_iris[['sepal length (cm)']]
y_iris = df_iris['petal length (cm)']

# Split the data into training and testing sets
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Create a simple linear regression model
model_iris = LinearRegression()
model_iris.fit(X_iris_train, y_iris_train)

# Make predictions on the test set
y_iris_pred = model_iris.predict(X_iris_test)

# Visualize the regression line
plt.scatter(X_iris_test, y_iris_test, color='black')
plt.plot(X_iris_test, y_iris_pred, color='blue', linewidth=3)
plt.title('Simple Linear Regression on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

# Interpretation
print(f"Intercept (beta_0): {model_iris.intercept_}")
print(f"Slope (beta_1): {model_iris.coef_[0]}")

# Evaluate model performance
print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_iris_test, y_iris_pred)}")
print(f"Mean Squared Error: {metrics.mean_squared_error(y_iris_test, y_iris_pred)}")
print(f"R-squared: {metrics.r2_score(y_iris_test, y_iris_pred)}")
