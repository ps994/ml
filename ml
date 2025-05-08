Naïve Bayes prediction model



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
emails = [
    "Win money now",          
    "Lowest price guaranteed",
    "Hey, are we still meeting tomorrow?",  
    "Let’s have lunch",       
    "Congratulations, you won!", 
    "Call me when you're free"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text data into numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict on new unseen data
new_emails = ["You won a prize", "Let’s meet for dinner"]
new_X = vectorizer.transform(new_emails)
predictions = model.predict(new_X)

# Output predictions
print("Predictions:", predictions)  # 1 = spam, 0 = not spam



Simple Linear Regression



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Sample Data (Years of Experience vs. Salary)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # Independent variable
y = np.array([30000, 35000, 40000, 45000, 50000, 60000])  # Dependent variable

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)

# Step 6: Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()




Multiple linear regression



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Sample Data (Features: sqft, bedrooms, age)
X = np.array([
    [1500, 3, 10],
    [1600, 3, 15],
    [1700, 4, 20],
    [1800, 4, 5],
    [2000, 5, 7],
    [2100, 4, 3],
    [2200, 5, 2]
])  # Independent variables

y = np.array([300000, 320000, 340000, 360000, 400000, 420000, 450000])  # House prices

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)  # [coef_sqft, coef_bedrooms, coef_age]

# Optional: Predict a new house price
new_house = np.array([[1900, 4, 8]])  # 1900 sqft, 4 bedrooms, 8 years old
predicted_price = model.predict(new_house)
print("Predicted price for new house:", predicted_price[0])



Polynomial Regression


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Sample non-linear data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([3, 6, 9, 15, 25, 40, 60, 85, 115])  # Non-linear relationship

# Step 2: Transform the features to polynomial features (e.g., degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Step 4: Predict
y_pred = model.predict(X_poly)

# Step 5: Evaluate the model
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 6: Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Polynomial Fit (Degree 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()




Lasso and Ridge Regression


import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Sample data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8]
])
y = np.array([5, 7, 9, 11, 13, 15, 17])  # Linear relationship: y = 1*x1 + 1*x2 + 2

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Lasso Regression (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Lasso Regression:")
print("Coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("R² Score:", r2_score(y_test, lasso_pred))
print()

# Step 4: Ridge Regression (L2)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("Ridge Regression:")
print("Coefficients:", ridge.coef_)
print("Intercept:", ridge.intercept_)
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("R² Score:", r2_score(y_test, ridge_pred))




Logistic regression


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Sample data (Hours studied vs. Pass/Fail)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Hours
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Step 2: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Predict new outcome
new_hours = np.array([[4.5]])
predicted_class = model.predict(new_hours)
predicted_prob = model.predict_proba(new_hours)

print("Prediction for 4.5 hours studied:", predicted_class[0])
print("Probability of passing:", predicted_prob[0][1])



K-NN classifier


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Sample data (feature1, feature2) with binary class labels
X = np.array([
    [1, 2], [2, 3], [3, 1],
    [6, 5], [7, 7], [8, 6]
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Class A, 1 = Class B

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the K-NN model
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print(f"K-NN Classifier (k={k}) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Predict a new sample
new_point = np.array([[5, 5]])
prediction = model.predict(new_point)
print("Prediction for new point [5, 5]:", prediction[0])


Decision tree classification


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Sample data (Weight, Texture: 0 = Smooth, 1 = Bumpy)
X = np.array([
    [150, 0],  # Apple
    [170, 0],  # Apple
    [140, 1],  # Orange
    [130, 1],  # Orange
])
y = np.array([0, 0, 1, 1])  # 0 = Apple, 1 = Orange

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Step 3: Train decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Predict new data
new_fruit = np.array([[160, 0]])  # Predict for a smooth fruit weighing 160g
prediction = model.predict(new_fruit)
print("Prediction for [160, 0]:", "Apple" if prediction[0] == 0 else "Orange")

# Step 7: Visualize the tree
plt.figure(figsize=(8, 6))
tree.plot_tree(model, feature_names=["Weight", "Texture"], class_names=["Apple", "Orange"], filled=True)
plt.title("Decision Tree for Fruit Classification")
plt.show()



SVM classification


import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Sample data (2D points with binary classes)
X = np.array([
    [2, 3], [1, 1], [2, 1], [3, 2],  # Class 0
    [7, 8], [8, 8], [9, 10], [8, 9]  # Class 1
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Train the SVM classifier (linear kernel)
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Predict new sample
new_point = np.array([[5, 5]])
prediction = model.predict(new_point)
print("Prediction for [5, 5]:", "Class 0" if prediction[0] == 0 else "Class 1")

# Step 7: Visualize decision boundary (only for 2D features)
def plot_svm_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # Plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_svm_decision_boundary(model, X, y)



K-means clustering


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic data with 2 features and 3 clusters
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Step 2: Apply K-Means clustering (k=3)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Step 3: Get the cluster centers and predicted cluster labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 4: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Step 5: Evaluate the clustering result
print("Cluster Centers:", centers)



Hierarchical clustering


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Generate synthetic data with 2 features and 3 clusters
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Step 2: Apply Agglomerative Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

# Step 3: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.title('Hierarchical Clustering (Agglomerative)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 4: Create a Dendrogram
linked = linkage(X, 'ward')  # 'ward' minimizes variance within clusters
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()


Artificial Neural Network


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Preprocess the data
# Normalize the features (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the labels to categorical (one-hot encoding)
y_one_hot = to_categorical(y)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.3, random_state=42)

# Step 4: Build the ANN model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))  # First hidden layer (8 neurons)
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes for Iris species)

# Step 5: Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot encoded predictions to class labels
y_test_classes = np.argmax(y_test, axis=1)  # Convert true labels to class labels

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))


