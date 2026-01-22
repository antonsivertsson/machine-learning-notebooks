import numpy as np
from sklearn.ensemble import IsolationForest

# Generate sample data (two-dimensional points)
#np.random.seed(0)
X = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack((X, X_outliers))

# Train the Isolation Forest model
clf = IsolationForest(contamination=0.1)
clf.fit(X)

# Predict anomalies
y_pred = clf.predict(X)

# Print the predicted labels (-1 for anomalies, 1 for inliers)
print("Predicted labels:")
print(y_pred)

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], color='b', label='Inliers')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='r', label='Anomalies')
plt.legend()
plt.title('Anomaly Detection with Isolation Forest')
plt.show()
