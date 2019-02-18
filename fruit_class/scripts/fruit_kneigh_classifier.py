#!/usr/bin/python -Wignore

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm

# Example of Supervised Learning - Classfication 

# Load data
fruits = pd.read_table('../data/fruit_data_with_colors.txt')


# Dataset to train system 
# color_score: 1-> Red, 0-> Violet
print (fruits.head())


# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
###print (lookup_fruit_name)

# The file contains the mass, height, and width of a selection of oranges, lemons and apples. The heights were measured along the core of the fruit. The widths were the widest width perpendicular to the height.

# ### Examining the data
# Always split data into Train and Test data to avoid our algoritm be only valid for our trainig dataset (X_train, X_test, y_train, y_test) 
# X is input parameters using which prediction is to be made (features)
# y is the output which is the prediction of fruit name (targets) fruit_label is the target label
# Following is the way to split the data into default 75%/25% Train-Test


# ### Create train-test split

# For this example, we use the mass, width, and height features of each fruit instance
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# plotting a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['mass'], X_train['width'], X_train['height'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('mass')
ax.set_ylabel('width')
ax.set_zlabel('height')
###plt.show()


# ### Create classifier object
# We will apply K-Nearest Neighbour algorithm of classification
knn = KNeighborsClassifier(n_neighbors = 5)


# ### Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)


# ### Estimate the accuracy of the classifier on future data, using the test data
print "Accuracy: " + str (knn.score(X_test, y_test))

# ### Use the trained k-NN classifier model to classify new, previously unseen objects

# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
lookup_fruit_name[fruit_prediction[0]]
print "Prediction for mass=20, width=4.3, height=5.5: " + str (lookup_fruit_name[fruit_prediction[0]])


# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
lookup_fruit_name[fruit_prediction[0]]

print "Prediction for mass=100, width=6.3, height=8.5: " + str (lookup_fruit_name[fruit_prediction[0]]) 


# ### How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?

k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);
###plt.show()


# ### How sensitive is k-NN classification accuracy to the train/test split proportion?

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');
###plt.show()
