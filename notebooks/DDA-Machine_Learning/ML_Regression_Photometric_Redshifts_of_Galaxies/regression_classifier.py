#!/usr/bin/python -Wignore


import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))

# paste your cross_validate_model function here
def cross_validate_model(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # initialise a list to collect median_diffs for each iteration of the loop below
  diffs = []

  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    model.fit(train_features, train_targets)
    
    # predict using the model
    predictions = model.predict(test_features)
 
    # calculate the median_diff from predicted values and append to results array
    diffs.append(median_diff(predictions, test_targets))
 
  # return the list with your median difference values
  return diffs

# complete this function
def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  galaxies = data[data['spec_class'] == b'GALAXY']
  qsos = data[data['spec_class'] == b'QSO']

  # return the seperated galaxies and qsos arrays
  return galaxies, qsos

def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]

    # fit the model for the current set
    model.fit(train_features, train_targets)

    # predict using the model
    predictions = model.predict(test_features)

    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions


def cross_validate_median_diff(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return np.mean(cross_validate_model(dtr, features, targets, 10))

if __name__ == "__main__":

  data = np.load('./sdss_galaxy_colors.npy')

  # split the data set into galaxies and QSOs
  galaxies, qsos= split_galaxies_qsos(data)

  #
  features_gal, targets_gal = get_features_targets(galaxies)
  features_qso, targets_qso = get_features_targets(qsos)

  #
  dtr = DecisionTreeRegressor(max_depth=19)

  #
  predictions_gal = cross_validate_predictions(dtr, features_gal, targets_gal, 10)
  predictions_qso = cross_validate_predictions(dtr, features_qso, targets_qso, 10)

  # here we cross validate the model and get the cross-validated median difference
  # the cross_validated_med_diff function is in "written_functions"
  galaxy_med_diff = cross_validate_median_diff(galaxies)
  qso_med_diff = cross_validate_median_diff(qsos)

  # print the results
  print("Median difference for Galaxies: {:.3f}".format(galaxy_med_diff))
  print("Median difference for QSOs: {:.3f}".format(qso_med_diff))

  # plot results
  plt.scatter(targets_gal, predictions_gal, s=0.4)
  plt.xlim((0, targets_gal.max()))
  plt.ylim((0, predictions_gal.max()))
  plt.xlabel('Measured Redshift for GAL')
  plt.ylabel('Predicted Redshift for GAL')
  plt.show()

  plt.scatter(targets_qso, predictions_qso, s=0.4)
  plt.xlim((0, targets_qso.max()))
  plt.ylim((0, predictions_qso.max()))
  plt.xlabel('Measured Redshift for QSO')
  plt.ylabel('Predicted Redshift for QSO')
  plt.show()

