import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w, h, d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=3, init='random',
            n_init=5, random_state=0)

km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)
colors = km.cluster_centers_[labels]
img = np.reshape(colors, (w, h, d))
plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

j = []

for n_clusters in range(1, 8):
    km = KMeans(n_clusters=n_clusters, init='k-means++',
                n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)
    j.append(km.inertia_)


plt.figure()
plt.plot(range(1, 8), j)
plt.show()

for i in range(0, 3):
    b = labels == i
    b = np.reshape(b, (w, h))
    plt.figure()
    plt.imshow(b)
    plt.tight_layout()
    plt.show()
