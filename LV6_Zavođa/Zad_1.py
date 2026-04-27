# U prilogu vježbe nalazi se funkcija 6.1. koja služi za generiranje umjetnih podataka kako bi se demonstriralo grupiranje podataka. 
# Funkcija prima cijeli broj koji definira željeni broju uzoraka u skupu i cijeli broj (od 1 do 5) koji definira na koji način će se generirati podaci, a vraća generirani skup podataka u obliku numpy polja 
# pri čemu su prvi i drugi stupac vrijednosti prve odnosno druge ulazne veličine za svaki podatak.
# Generirajte 500 podataka i prikažite ih na slici. Pomoću scikit-learn ugrađene metode za kmeans odredite centre klastera te svaki podatak obojite ovisno o njegovoj pripadnosti pojedinom klasteru (grupi).
#  Nekoliko puta pokrenite napisani kod. Što primjećujete? Što se događa ako mijenjate način kako se generiraju podaci?
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

X = generate_data(n_samples=500, flagc=1) 

km = KMeans(n_clusters=3, n_init=10)
labels = km.fit_predict(X)
centers = km.cluster_centers_
 
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red') 
plt.title("Kmeans grupiranje podataka)")
plt.show()