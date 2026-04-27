# Primijenite scikit-learn kmeans metodu za kvantizaciju boje na slici. Proučite kod 6.2. iz priloga vježbe te ga primijenite za kvantizaciju boje na slici example_grayscale.png 
# koja dolazi kao prilog ovoj vježbi. Mijenjajte broj klastera. Što primjećujete? Izračunajte kolika se kompresija ove slike može postići ako se koristi 10 klastera.

import matplotlib.image as mpimg
import scipy as sp
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt

imageNew = mpimg.imread('example_grayscale.png')
X = imageNew.reshape((-1, 1)) 
k_means = cluster.KMeans(n_clusters=10,n_init=1)
k_means.fit(X) 

values = k_means.cluster_centers_.squeeze()
labels = k_means.fit_predict(X)
compressed_image = np.choose(labels, values).reshape(imageNew.shape)

plt.figure(1)
plt.imshow(compressed_image, cmap='gray')
plt.show()



