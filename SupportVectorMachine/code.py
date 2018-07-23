from SupportVectorMachine import SVM
import numpy  as np

features = np.array([[1,7], [2,8], [3,8], [5,1], [6,-1], [7,3]])
labels = np.array([-1,-1,-1,1,1,1])

clf = SVM()
clf.fit(features, labels)

predict_us = [[0,10],[1,3],[3,4],[3,5]]
for p in predict_us:
    print(p, clf.predict(p))
