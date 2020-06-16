"""
ML
1. Load Library : Depedencies
2. Load Dataset : iris, est ...
3. Split the dataset into train and test
4. Train model : KNN, K-Means, est ...
5. Prediction
Save and Load:
1. import joblib
2. joblib.dump -> Save
3. joblib.load -> Load
"""
#Load Library
import numpy as np
# Load dataset 
from sklearn.datasets import load_iris 
iris = load_iris() 
X = iris.data 
y = iris.target 


# Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, random_state = 2020) 

# import KNN model
from sklearn.neighbors import KNeighborsClassifier as KNN 
knn = KNN(n_neighbors = 5) 

# train model 
knn.fit(X_train, y_train) 
# Prediction
print("Before save:",knn.predict(X_test))
from sklearn.externals import joblib 

# Save the model
joblib.dump(knn, 'my_model_knn.pkl') 

# Load the model from the file 
knn_from_joblib = joblib.load('my_model_knn.pkl') 

# Use the loaded model to make predictions 
print("After save:",knn_from_joblib.predict(X_test))
