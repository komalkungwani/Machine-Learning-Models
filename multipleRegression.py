import pandas as pd
import numpy as np

"""
Formula for multiple linear regression is
y = b0x0 + b1x1 + b2x2 + b3x3....
here b0, b1, b2, b3 are constants
x0 = 1
x1, x2, x3 are the features
converting it to a matrix we get
y = Xb + e
X = [ [1, x01, x02, x03...x0p], 
      [1, x11, x12, x13...x1p]......
      [1, xp1, xp2, xp3....xpp]]
  
b = [ b0
      b1 
      b2 
      b3
    ]

e = [ e0
      e1
      e2
      e3
    ]
now we determine estimate of b using least square method
b' = (Xt.X)^-1.Xty
here Xt means transpose of X and ^-1 means inverse
hence 
y' = Xb'
"""

class Multiple:
    def __init__(self):
        self.features = []
        self.a = np.array([1])
"""
fit function has same meaning as in sklearn module same is with predict function     
"""
    def fit(self, features, labels):
        for i in range(len(features)):
            self.features.append(np.concatenate((self.a, features[i]), axis=0))
        self.features = np.array(self.features)
        featuresT = self.features.transpose()
        constants = np.matmul(featuresT, self.features)
        constants = np.linalg.inv(constants)
        constants = np.dot(constants, featuresT)
        self.constants = np.dot(constants, labels)
        y_cap = np.dot(self.features, self.constants)
        self.e = np.subtract(labels, y_cap)
        self.e = np.mean(self.e)

    def predict(self, feature):
        result = []
        for feature in features:
            featureP = np.concatenate((self.a, feature), axis=0)
            result.append((np.dot(featureP, self.constants)))
        return result
