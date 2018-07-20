import operator
import numpy as np

class KNN:
    def __init__(self,k):
        self.k =k
    
    def __get_neighbours__(self, train_set, test_set):
        dist = np.sqrt(np.sum((train_set - test_set)**2 , axis=1))
        return np.argsort(dist)[0:self.k]

    
    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        
        
    def predict(self, test_features):
        # get the index of all nearest neighbouring data points
        neighbors = self.__get_neighbours__(self.features, test_features)
        classVotes = {}
        # to count votes for each class initialise all class with zero votes
        for label in np.unique(self.labels):
             classVotes[label] = 0
        # add count to class that are present in the nearest neighbors data points
        for index in neighbors:
            closest_lable = self.labels[index]
            classVotes[closest_lable[0]] += 1

        return max(classVotes.items(), key = operator.itemgetter(1))[0]