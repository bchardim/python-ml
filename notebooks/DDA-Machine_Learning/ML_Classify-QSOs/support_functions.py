import numpy as np
import itertools
from matplotlib import pyplot as plt



def generate_features_targets(data):
    output_targets = np.empty(shape=(len(data)), dtype='<U20')
    
    ####
    # Once tested our model with training file, we proceed to predict values
    # When using sdssdr6_colors_class.200000.npy, file to predict no info about z only to kind of objects (0,1) classified
    # !!! Features are only colors, so prediction is based only by spectrum color !!! Once model has been trained !!!
    # data['label']=0 -> QSOs, there are around 13000 QSOs and 187000 Stars (True value)
    output_targets[:] = data['label']
    ###

    ###
    # When using training file sdssdr6_colors_class_train.npy, file to test our model
    # z>0 -> QSO
    ###for (i, item) in enumerate(data['redshift']):
    ###    if item > 0:
    ###        output_targets[i] = 'QSO'
    ###    else:
    ###        output_targets[i] = 'Star' 
    ###

    input_features = np.empty(shape=(len(data), 4))
    input_features[:, 0] = data['u-g']
    input_features[:, 1] = data['g-r']
    input_features[:, 2] = data['r-i']
    input_features[:, 3] = data['i-z']
    
    return input_features, output_targets



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
