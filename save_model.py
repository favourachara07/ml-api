import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load the Wisconsin Breast Cancer Dataset
data = datasets.load_breast_cancer()

# Create a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names) # seting features dataframe
df['target'] = data.target #setting target dataframe

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = svm.SVC(kernel='linear', probability=True)

# Train the SVM model
svm_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model using accuracy, precision, recall, and F1 score
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)

joblib.dump(svm_classifier, 'svm_model.pkl')
# Save the model and scaler
joblib.dump(scaler, 'scaler.pkl')  # Save scaler if you used feature scaling