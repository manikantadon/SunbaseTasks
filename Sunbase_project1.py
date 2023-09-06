import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Data Preprocessing
# Load the dataset
file_path = "your_dataset.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Initial data exploration
print(data.head())
print(data.info())

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Location'], drop_first=True)  # One-hot encoding for 'Location'

# Split the data into training and testing sets
X = data.drop(columns=['CustomerID', 'Name', 'Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Engineering
# Feature scaling (if necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Building
# Choose an appropriate machine learning algorithm (Random Forest)
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Step 4: Model Optimization
# Fine-tune the model parameters using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Step 5: Model Evaluation
# Validate the model on the testing dataset
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Step 6: Model Deployment (Simulated)
# In a real deployment, you would use a framework like Flask for deployment.
# Here, we'll simulate model deployment by making predictions on new data.

new_customer_data = pd.DataFrame({
    'Age': [40],
    'Gender': ['Female'],
    'Subscription_Length_Months': [10],
    'Monthly_Bill': [70.0],
    'Total_Usage_GB': [300],
    'Location_Los Angeles': [1],
    'Location_Miami': [0],
    'Location_New York': [0],
    'Location_Houston': [0],
    'Location_Chicago': [0]
})

# Encode categorical variables in the new data
new_customer_data['Gender'] = label_encoder.transform(new_customer_data['Gender'])

# Scale the features
new_customer_data_scaled = scaler.transform(new_customer_data)

# Make churn prediction for the new customer
churn_prediction = best_model.predict(new_customer_data_scaled)

if churn_prediction[0] == 0:
    print("New customer is not likely to churn.")
else:
    print("New customer is likely to churn.")

