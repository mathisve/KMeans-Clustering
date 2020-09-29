import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class KMeans():
  def __init__(self, n_clusters, data):
    self.k = n_clusters
    self.x = data
    self.d = self.x[0].shape[0] # Dimension
    self.centroids = np.zeros((self.d,self.k))

    # Initializing K centroids within the data bounds
    # Respecting the diffirent max and min of the X and Y axis
    for i in range(0, self.d):
        self.centroids[i] = np.random.uniform(low=np.min(self.x[:, i]), high=np.max(self.x[:, i]), size=(1, self.k))
    self.centroids = np.transpose(self.centroids)

    self.labels = self.genLabels()

    self.colors = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'purple', 5:'orange', 6:'brown'}

  def dist(self, p, q):
    # Euclidian distance
    return math.sqrt(np.sum(np.power(np.subtract(p, q), 2)))

  def predict(self, p):
    return np.argmin([self.dist(p, centroid) for centroid in self.centroids])

  def genLabels(self):
    return np.asarray([self.predict(p) for p in self.x])

  def fit(self, epochs, step):
    print(f"\nFitting for {epochs} epochs with step of {step}")
    for epoch in range(0, epochs):
      for i, centroid in enumerate(self.centroids):

        # VECTORS!!
        l = [1 if item == i else 0 for item in self.labels]
        temp = np.matmul(np.transpose(np.subtract(x, centroid)), np.transpose(l))
        self.centroids[i] += np.matmul([step], np.divide(temp.reshape(1,2), sum(l)))

        # Regenerate the labels
        self.labels = self.genLabels()

    print("Done fitting!\n")

  def showPlot(self, classification=False):
    plt.figure(1, figsize = (10,10))

    if classification:
      for i, p in enumerate(self.x):
       plt.scatter(p[0], p[1], color=self.colors[self.labels[i]])

      for i, centroid in enumerate(self.centroids):
        plt.scatter(centroid[0], centroid[1], color=self.colors[i], s=100, marker="x")

    else:
      plt.scatter(self.x[:,0], self.x[:,1], color='blue')

      plt.scatter(self.centroids[:,0], self.centroids[:,1], color="red", s=100)

    plt.show()

  def printCentroids(self):
    print("="*20)
    print("Centroids:")
    print(self.centroids)
