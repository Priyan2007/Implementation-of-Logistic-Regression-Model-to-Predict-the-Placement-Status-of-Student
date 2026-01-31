# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the student placement dataset and preprocess it by removing irrelevant columns and encoding categorical values.
2. Split the dataset into training and testing sets and apply feature scaling to the input data. 
3. Train a Logistic Regression model using the training dataset.
4. Predict the placement status using test data and evaluate the model using accuracy and performance metrics.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRIYAN V
RegisterNumber:  212224230211
*/
```

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Placement_Data.csv")
print("First 5 rows of the dataset:\n")
print(data.head())

if "sl_no" in data.columns:
    data = data.drop("sl_no", axis=1)

if "salary" in data.columns:
    data = data.drop("salary", axis=1)   

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

X = data.drop("status", axis=1)
y = data["status"]               

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not Placed", "Actual: Placed"],
    columns=["Predicted: Not Placed", "Predicted: Placed"]
)

print("\nConfusion Matrix:")
print(cm_df)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\nClassification Report:")
print(report_df.round(2))
```

## Output:

<img width="761" height="539" alt="image" src="https://github.com/user-attachments/assets/8de92960-29f3-4f07-bf4c-08813ad856e8" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
