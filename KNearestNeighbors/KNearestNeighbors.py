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
        results = []
        for test_feature in test_features:
            # get the index of all nearest neighbouring data points
            neighbors = self.__get_neighbours__(self.features, test_feature)
            classVotes = {}
            # to count votes for each class initialise all class with zero votes
            for label in np.unique(self.labels):
                 classVotes[label] = 0
            # add count to class that are present in the nearest neighbors data points
            for index in neighbors:
                closest_lable = self.labels[index]
                classVotes[closest_lable[0]] += 1
            results.append(max(classVotes.items(), key = operator.itemgetter(1))[0])

        return results
    
    def accuracy(self, y_test, predictions):
        correct = 0
        for i in range(len(y_test)):
            if(y_test[i] == predictions[i]):
                correct += 1
                
        return correct/len(y_test)
