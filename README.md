# KMeans-Clustering :milky_way:
Very simple and easy to use vectorized KMeans clustering algorithm from scratch in Python.
Now supports **more than 2 dimensions!**

## Ease of use: :tada:
```python
kmeans = KMeans(n_clusters=4, data=x)
kmeans.fit(epochs=100, steps=0.1)
pred = kmeans.predict([-2, 3])
print(kmeans.colors[pred])
```

## Resuts: :chart_with_upwards_trend:

### **K = 2**

![gif1](https://github.com/Mathisco-01/KMeans-Clustering/blob/master/images/cluster2.gif?raw=true)

### **K = 4**

![gif2](https://github.com/Mathisco-01/KMeans-Clustering/blob/master/images/cluster4.gif?raw=true)
