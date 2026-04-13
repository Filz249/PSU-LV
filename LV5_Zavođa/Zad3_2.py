
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
 
df = pd.read_csv("occupancy_processed.csv")
 
x = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']
 
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
 
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
 
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(x_train_scaled, y_train)
 
y_pred = dt.predict(x_test_scaled)
 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Slobodna', 'Zauzeta'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica zabune - stablo odlučivanja")
plt.show()
 
accuracy = accuracy_score(y_test, y_pred)
print("Točnost:", accuracy)
 
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))
 
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
 
print("Preciznost po klasama:", precision)
print("Odziv po klasama:", recall)
 
plt.figure(figsize = (14, 8))
plot_tree(
    dt,
    feature_names = ['S3_Temp', 'S5_CO2'],
    class_names = ['Slobodna', 'Zauzeta'],
    filled=True
)
plt.title("Stablo odlučivanja")
plt.show()
 
for depth in [1, 2, 3, 5, 7, 10, None]:
    dt = DecisionTreeClassifier(max_depth = depth, random_state = 42)
    dt.fit(x_train_scaled, y_train)
    y_pred = dt.predict(x_test_scaled)
 
    print(f"\n=== max_depth = {depth} ===")
    print("Točnost:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names = ['Slobodna', 'Zauzeta']))
 
 