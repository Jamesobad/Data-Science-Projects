import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample data
data = {
    'Age': [25, 34, 28, 45, 38, 50, 22, 60, 29, 40],
    'Height': [165, 160, 170, 175, 160, 180, 155, 165, 168, 172],
    'Weight': [70, 85, 68, 95, 78, 100, 50, 90, 72, 88],
    'BMI': [25.7, 33.2, 23.5, 31.0, 30.5, 30.9, 20.8, 33.0, 25.5, 29.7],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Physical Activity': [3, 1, 5, 2, 1, 0, 6, 1, 4, 2],
    'Diet Quality': [3, 2, 4, 2, 3, 1, 5, 2, 4, 3],
    'Sleep Hours': [7, 6, 8, 5, 6, 4, 9, 5, 7, 6],
    'Obese': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
xtrain, xtest, ytrain, ytest = train_test_split(df[['Age', 'Height', 'Weight', 'BMI', 'Physical Activity', 'Diet Quality', 'Sleep Hours']], df['Obese'], test_size=0.3)
model = LogisticRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy)
p1={
    'Age':[22],
    'Height': [170],
    'Weight': [100],
    'BMI': [40.5],
    'Gender': ['Male'],
    'Physical Activity': [5],
    'Diet Quality': [3],
    'Sleep Hours': [13],
    'Obese': [1]
}
df1=pd.DataFrame(p1)
prediction = model.predict(df1[['Age', 'Height', 'Weight', 'BMI', 'Physical Activity', 'Diet Quality', 'Sleep Hours']])
print("Prediction:", prediction)
accuracy = accuracy_score(df1['Obese'], prediction)
print("Accuracy:", accuracy)
