import numpy as np

class Kmeans:
    def __init__(self, k=3, tolerance = 0.0001, max_iterations = 500):
        #k: number of clusters to form
        self.k = k
        #tolerance: minimum difference between two centroids
        self.tol=tolerance
        #max_iterations: maximum number of iterations to run
        self.max_iter = max_iterations

    def fit(self, features):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = features[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in features:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,features):
        distances = [np.linalg.norm(features-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
