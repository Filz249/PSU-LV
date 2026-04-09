import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
 
def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) \
        - 1.1622*np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y
 
def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1 * varNoise * np.random.normal(0, 1, len(y))
    return y_noisy
 
def run_experiment(n_samples=50, degrees=[2, 6, 15]):
    print(f'\n===== Broj uzoraka: {n_samples} =====')
 
    x = np.linspace(1, 10, n_samples)
    y_true = non_func(x)
    y_measured = add_noise(y_true)
 
    x = x[:, np.newaxis]
    y_measured = y_measured[:, np.newaxis]
 
    np.random.seed(12)
    indeksi = np.random.permutation(len(x))
    split = int(np.floor(0.7 * len(x)))
    indeksi_train = indeksi[:split]
    indeksi_test = indeksi[split:]
 
    MSEtrain = []
    MSEtest = []
 
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'k--', linewidth=2, label='pozadinska funkcija')
 
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)
 
        xtrain = x_poly[indeksi_train]
        ytrain = y_measured[indeksi_train]
        xtest = x_poly[indeksi_test]
        ytest = y_measured[indeksi_test]
 
        model = lm.LinearRegression()
        model.fit(xtrain, ytrain)
 
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)
        y_all_pred = model.predict(x_poly)
 
        mse_train = mean_squared_error(ytrain, ytrain_pred)
        mse_test = mean_squared_error(ytest, ytest_pred)
 
        MSEtrain.append(mse_train)
        MSEtest.append(mse_test)
 
        plt.plot(x, y_all_pred, label=f'degree={degree}')
 
    plt.plot(x[indeksi_train], y_measured[indeksi_train], 'ob', alpha=0.7, label='train')
    plt.plot(x[indeksi_test], y_measured[indeksi_test], 'or', alpha=0.7, label='test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Usporedba modela za n_samples = {n_samples}')
    plt.legend()
    plt.grid(True)
    plt.show()
 
    print('MSEtrain =', MSEtrain)
    print('MSEtest  =', MSEtest)
 
    return MSEtrain, MSEtest
 
# osnovni slučaj
MSEtrain, MSEtest = run_experiment(n_samples=50, degrees=[2, 6, 15])
 
# simulacija: manje uzoraka
MSEtrain_small, MSEtest_small = run_experiment(n_samples=20, degrees=[2, 6, 15])
 
# simulacija: više uzoraka
MSEtrain_large, MSEtest_large = run_experiment(n_samples=200, degrees=[2, 6, 15])