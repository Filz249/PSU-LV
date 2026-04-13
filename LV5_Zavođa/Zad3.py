import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
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

tree = DecisionTreeClassifier(max_depth = 4, random_state = 42)
tree.fit(x_train_scaled, y_train)

y_pred = tree.predict(x_test_scaled)

print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost:", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred, average = None))
print("Odziv:", recall_score(y_test, y_pred, average = None))

plt.figure(figsize=(14,7))
plot_tree(
    tree,
    feature_names = feature_names,
    class_names = class_names,
    filled = True
)
plt.title("Stablo odlučivanja")
plt.show()


dt = DecisionTreeClassifier()
dt.fit(x, y)

for d in range(1, 11):
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(x_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(x_test_scaled))
    print(f"max_depth={d} -> točnost={acc:.4f}")

tree_raw = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_raw.fit(x_train, y_train)

y_pred_raw = tree_raw.predict(x_test)

print("\nBez skaliranja")
print("Točnost bez skaliranja:", accuracy_score(y_test, y_pred_raw))