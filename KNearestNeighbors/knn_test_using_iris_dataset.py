from KNearestNeighbors import KNN
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.txt')

X = df.iloc[:,[0,1,2,3]].values
y = df.iloc[:,[4]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
 
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(clf.accuracy(y_test, predictions))
