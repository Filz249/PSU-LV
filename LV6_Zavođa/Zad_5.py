from sklearn import cluster
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread("example.png")

if image.shape[2] == 4:
    image = image[:, :, :3]

h, w, c = image.shape
X = image.reshape((-1, c))

k_means = cluster.KMeans(n_clusters=10, n_init=10)
k_means.fit(X)

values = k_means.cluster_centers_
labels = k_means.labels_

image_compressed = values[labels].reshape(image.shape)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Originalna slika")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image_compressed)
plt.title("Kvantizirana slika")
plt.axis("off")

plt.show()