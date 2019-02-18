#!/usr/bin/python -Wignore

import numpy as np
import itertools
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')



if __name__ == "__main__":

    # Example of Supervised Learning - Classfication 
    # Load data
    fruits = pd.read_table('../data/fruit_data_with_colors.txt')

    # Dataset to train system 
    # color_score: 1-> Red, 0-> Violet
    print (fruits.head())

    # create a mapping from fruit label value to fruit name to make results easier to interpret
    lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

    # For this example, we use the mass, width, and height features of each fruit instance
    X = fruits[['mass', 'width', 'height', 'color_score']]
    y = fruits['fruit_label']

    # default is 75% / 25% train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # instantiate a random forest classifier
    # A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples 
    # of the dataset and use averaging to improve the predictive accuracy and control over-fitting. 
    #
    # We can improve the accuracy of our classification by using a collection (or ensemble) of trees as known as a random forest
    # When making a prediction, every tree in the forest gives its own prediction and the most common classification is taken as the overall forest prediction
    # The random forest is around ~7% more accurate than a standard decision tree.
    rfc = RandomForestClassifier(n_estimators=50)

    # get predictions using 10-fold cross validation with cross_val_predict
    # Generate cross-validated estimates for each input data point
    #
    # In K Fold cross validation, the data is divided into k subsets. Now the holdout method is repeated k times, such that each time, one of the k subsets is used as the 
    # test set/ validation set and the other k-1 subsets are put together to form a training set. The error estimation is averaged over all k trials to get total effectiveness of our model.
    y_train_pred = cross_val_predict(rfc, X_train, y_train, cv=15)

    # calculate the model score 
    model_score = accuracy_score(y_train, y_train_pred) 
    print "Our accuracy score: " +  str(model_score)

    # Show predictions


    # Show decision tree


    # calculate the models confusion matrix using sklearns confusion_matrix function
    # 
    # The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix.
    # The confusion matrix shows the ways in which your classification model is confused when it makes predictions.
    # In addition to an overall accuracy score, we'd also like to know where our model is going wrong.
    # The x axis represents the predicted classes and the y axis represents the correct classes. The value in each cell is the number of examples with those predicted and actual classes.
    # Correctly classified objects are along the diagonal of the matrix

    class_labels = lookup_fruit_name.values()
    model_cm = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

    # Plot the confusion matrix using the provided functions.
    plt.figure()
    plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
    plt.show()

