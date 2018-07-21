# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from multipleRegression import Multiple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Fitting Created Multiple Linear Regression to the Training set
regressor = Multiple()
regressor.fit(X_train, y_train)

# Predicting the Test set results using sklearn r2_score
y_pred = regressor.predict(X_test)
print(r2_score(y_test, y_pred))
