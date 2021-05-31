## OS
import os
import sys


## Maths
import numpy as np
import random


## Graphics
import matplotlib.pyplot as plt


## DataFrames
import pandas as pd


## Sklearn
# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
# metrics
from sklearn.metrics import roc_auc_score, accuracy_score


## Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda, Dropout, BatchNormalization, Activation, concatenate, subtract
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers




def plot_var(variable, history):
    plt.title(variable)
    plt.plot(history.history[variable][2:], label='train')
    plt.plot(history.history['val_'+variable][2:], label='validation')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel(variable)
    plt.grid(True)


def make_train_validation_test_triplets_list(triplet_file):

    triplets = np.loadtxt(triplet_file)

    train_triplets_file = "./train_triplets_list.txt"
    validation_triplets_file = "./validation_triplets_list.txt"
    test_triplets_file = "./test_triplets_list.txt"

    if os.path.exists(train_triplets_file) and os.path.exists(validation_triplets_file) and os.path.exists(test_triplets_file):
        triplets_train = np.loadtxt(train_triplets_file)
        triplets_validation = np.loadtxt(validation_triplets_file)
        triplets_test = np.loadtxt(test_triplets_file)
        
    else:
        train_images = random.sample(range(0, 5000), 3400)  #list(range(0, 3800))

        triplets_train = [ t for t in triplets if (t[0] in train_images     and t[1] in train_images     and t[2] in train_images)     ]
        triplets_vt    = [ t for t in triplets if (t[0] not in train_images and t[1] not in train_images and t[2] not in train_images) ]

        triplets_validation, triplets_test = train_test_split(triplets_vt, train_size=0.5)

        np.savetxt(train_triplets_file, triplets_train)
        np.savetxt(validation_triplets_file, triplets_validation)
        np.savetxt(test_triplets_file, triplets_test)

    print("Train dataset size:      %d" %(len(triplets_train)))
    print("Validation dataset size: %d" %(len(triplets_validation)))
    print("Test dataset size:       %d" %(len(triplets_test)))


    return triplets_train, triplets_validation, triplets_test
    

def make_triplets_0(triplets):

    return [ [t[0], t[2], t[1]} for t in triplets ]


def make_triplets_from_file(array, triplet_file):

    triplets = np.loadtxt(triplet_file)
    data = np.array([ np.array([ array[int(x[0]), :], array[int(x[1]), :], array[int(x[2]), :] ]) for x in triplets ])

    return data
 

def not_(x):
    return -x+1


def make_triplets_and_target(triplet_file):
    ## Make train and validation triplets making sure there are no common images in the train and validation triplets
    triplets_train_1, triplets_validation_1, triplets_test_1 = make_train_validation_test_triplets_list(triplet_file)
    triplets_train_0      = make_triplets_0(triplets_train_1)
    triplets_validation_0 = make_triplets_0(triplets_validation_1)
    triplets_test_0       = make_triplets_0(triplets_test_1)
    
    triplets_train_1_in, triplets_train_1_out, triplets_train_0_out, triplets_train_0_in                     = train_test_split(triplets_train_1, triplets_train_0, train_size=0.5)
    triplets_validation_1_in, triplets_validation_1_out, triplets_validation_0_out, triplets_validation_0_in = train_test_split(triplets_validation_1, triplets_validation_0, train_size=0.5)
    triplets_test_1_in, triplets_test_1_out, triplets_test_0_out, triplets_test_0_in                         = train_test_split(triplets_test_1, triplets_test_0, train_size=0.5)

    ## Output variable
    n1 = len(triplets_train_1_in)
    n0 = len(triplets_train_0_in)
    y_train = np.array( n1*[1.] + n0*[0.] )
    y_2D_train = np.array(list(map(list, zip(y_train, not_(y_train)))))

    n1 = len(triplets_validation_1_in)
    n0 = len(triplets_validation_0_in)
    y_validation = np.array( n1*[1.] + n0*[0.] )
    y_2D_validation = np.array(list(map(list, zip(y_validation, not_(y_validation)))))

    n1 = len(triplets_test_1_in)
    n0 = len(triplets_test_0_in)
    y_test = np.array( n1*[1.] + n0*[0.] )
    y_2D_test = np.array(list(map(list, zip(y_test, not_(y_test)))))

    return triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in, y_2D_train, y_2D_validation, y_2D_test



def main():

    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "./datasets/train_triplets.txt"

    triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in, y_2D_train, y_2D_validation, y_2D_test = make_triplets_and_target(triplet_file):

    ## Make model
    model = create_model(...)
    
    print("Model summary:")
    print(model.summary())


    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['acc'],
    )
   


    ## Loop
    # ...

    print("\nFitting...")
    #history = model.fit((X_train[:, 0], X_train[:, 1], X_train[:, 2]), y_2D_train,
    #                    validation_data=((X_validation[:, 0], X_validation[:, 1], X_validation[:, 2]), y_2D_validation),
    #                    epochs=params["n_epochs"],
    #                    batch_size=params["batch_size"])


    ### Prediction on the test dataset
    #print("\nPredictions for the test sample...")
    #y_pred_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    #auc = roc_auc_score(y_2D_test[:, 0], y_pred_proba[:, 0])
    #print("ROC AUC: %.2f" %auc)

    #cuts = np.logspace(-2, 1, 100)
    #y_preds = [ y_pred_proba[:, 0] >= cut for cut in cuts ]
    #acc_scores = [ accuracy_score(y_2D_test[:, 0], y_preds[icut]) for icut in range(len(cuts)) ]


    ##### Accuracy as a fct of cut
    ##plt.figure()
    ##plt.plot(cuts, acc_scores)
    ##plt.xlabel("Cut value")
    ##plt.ylabel("Accuracy")
    ##plt.xscale("log")
    ##plt.savefig("acc_score.pdf")
    ##plt.close()


    #### Best accuracy cut
    #best_cut_idx = np.argmax(acc_scores)
    #best_cut = cuts[best_cut_idx]
    ##print("Best cut: %.3f" %best_cut)
    #y_pred = y_preds[best_cut_idx]

    #if abs(best_cut-0.5)>0.1:
    #    print("WARNING: Check best cut!")
    #    print("Best cut: %.3f" %best_cut)

    #best_cut = 0.5
    #y_pred = y_pred_proba[:, 0] >= best_cut

    #accuracy = accuracy_score(y_2D_test[:, 0], y_pred)
    #print("Accuracy: %.3f" %accuracy)


    ### Control plots
    ## Loss
    #for variable in ("loss", "acc"):
    #    plt.figure()
    #    plot_var(variable, history)
    #    plt.savefig(variable + ".pdf")
    #    plt.close()


    #del X_train, X_validation

    #return accuracy, auc

    ### Load test dataset
    #print("\nPredictions for the test dataset...")
    #triplet_file = "./datasets/test_triplets.txt"
    #X_test = make_triplets_from_file(train_array, triplet_file)
    #y_pred_test_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    #y_pred_test = y_pred_test_proba[:, 0] >= best_cut

    #np.savetxt("submit.txt", y_pred_test, fmt="%d")


