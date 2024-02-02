import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the Twomer dataset
input_file_path = '/content/drive/MyDrive/Drebin Dataset.csv'
df = pd.read_csv(input_file_path)

# Map class labels to binary values
class_mapping = {'B': 0, 'S': 1}
df['class'] = df['class'].map(class_mapping)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop('class', axis=1)
y = df['class']

# Use SimpleImputer to replace missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

print("Model Build Successful")
