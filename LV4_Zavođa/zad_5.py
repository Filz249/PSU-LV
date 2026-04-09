import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error
 
# ucitaj podatke
df = pd.read_csv('cars_processed.csv')
 
# 1. izbaci nepotrebne stupce i zadrži samo numeričke
df_num = df.drop(columns=['name'], errors='ignore')
df_num = df_num.select_dtypes(include=np.number)
 
print('Numerički stupci:')
print(df_num.columns.tolist())
 
# ulazi i izlaz
X = df_num.drop('selling_price', axis=1)
y = df_num['selling_price']
 
# 2. train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# 3A. skaliranje - StandardScaler
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)
 
# 4. model
model_std = LinearRegression()
model_std.fit(X_train_std, y_train)
 
# 5. evaluacija
y_train_pred_std = model_std.predict(X_train_std)
y_test_pred_std = model_std.predict(X_test_std)
 
print('\nStandardScaler ')
print('MAE train:', mean_absolute_error(y_train, y_train_pred_std))
print('MAE test :', mean_absolute_error(y_test, y_test_pred_std))
print('MSE train:', mean_squared_error(y_train, y_train_pred_std))
print('MSE test :', mean_squared_error(y_test, y_test_pred_std))
print('R2 train :', r2_score(y_train, y_train_pred_std))
print('R2 test  :', r2_score(y_test, y_test_pred_std))
print('Max error test:', max_error(y_test, y_test_pred_std))
 
# 3B. skaliranje - MinMaxScaler
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm = mm_scaler.transform(X_test)
 
model_mm = LinearRegression()
model_mm.fit(X_train_mm, y_train)
 
y_train_pred_mm = model_mm.predict(X_train_mm)
y_test_pred_mm = model_mm.predict(X_test_mm)
 
print('\nMinMaxScaler')
print('MAE train:', mean_absolute_error(y_train, y_train_pred_mm))
print('MAE test :', mean_absolute_error(y_test, y_test_pred_mm))
print('MSE train:', mean_squared_error(y_train, y_train_pred_mm))
print('MSE test :', mean_squared_error(y_test, y_test_pred_mm))
print('R2 train :', r2_score(y_train, y_train_pred_mm))
print('R2 test  :', r2_score(y_test, y_test_pred_mm))
print('Max error test:', max_error(y_test, y_test_pred_mm))
 
# 6. što se događa kad mijenjamo broj ulaznih varijabli
print('\nUtjecaj broja ulaznih veličina')
results = []
 
feature_sets = [
    ['year', 'km_driven'],
    ['year', 'km_driven', 'engine', 'max_power'],
    [col for col in X.columns]
]
 
for features in feature_sets:
    X_sub = df_num[features]
    y_sub = df_num['selling_price']
 
    Xtr, Xte, ytr, yte = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
 
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
 
    model = LinearRegression()
    model.fit(Xtr_s, ytr)
 
    ypred = model.predict(Xte_s)
 
    results.append({
        'features': features,
        'n_features': len(features),
        'MAE_test': mean_absolute_error(yte, ypred),
        'MSE_test': mean_squared_error(yte, ypred),
        'R2_test': r2_score(yte, ypred)
    })
 
for r in results:
    print('\nBroj ulaznih veličina:', r['n_features'])
    print('Varijable:', r['features'])
    print('MAE test:', r['MAE_test'])
    print('MSE test:', r['MSE_test'])
    print('R2 test:', r['R2_test'])
 
# 6. Što se događa s pogreškom na testnom skupu kada mijenjate broj ulaznih veličina?
# smanjuje se