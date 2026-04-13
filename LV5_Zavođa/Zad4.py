import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'

x = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

class_names = ['Slobodna', 'Zauzeta']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    stratify = y,
    random_state = 42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

log_reg = LogisticRegression()
log_reg.fit(x_train_scaled, y_train)

y_pred = log_reg.predict(x_test_scaled)

print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost:", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred, average = None))
print("Odziv:", recall_score(y_test, y_pred, average = None))