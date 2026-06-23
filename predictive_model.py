import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset.csv")

print("Dataset")
print(df)

# Features and Target
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE:", mae)
print("R2 Score:", r2)

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.savefig("graphs/prediction_graph.png")
plt.close()

# Sample Prediction
new_student = [[8, 90]]
predicted_marks = model.predict(new_student)

print("\nPredicted Marks for Student:")
print(predicted_marks[0])

print("\nProject Completed Successfully")