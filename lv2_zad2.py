import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)
mpg = data[:, 0]
hp = data[:, 3]
wt = data[:, 5]
plt.scatter(mpg, hp, wt, marker = '.')
plt.xlabel('mpg')
plt.ylabel('hp')
plt.show()
print(mpg.min())
print(mpg.max())
print(mpg.mean())