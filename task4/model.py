## OS
import os
import sys


## Maths
import numpy as np


## Graphics
import matplotlib.pyplot as plt


## DataFrames
import pandas as pd


## Sklearn
# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# metrics
from sklearn.metrics import roc_auc_score, accuracy_score


## Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Activation, concatenate, subtract
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



features_directory = "preprocessed_features/"


def plot_var(variable, history):
    plt.title(variable)
    plt.plot(history.history[variable][2:], label='train')
    plt.plot(history.history['val_'+variable][2:], label='validation')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel(variable)
    plt.grid(True)



 
def int2key(x):
    return "%05d" %x


#def make_triplets(df, triplet_file):
#
#    features = [ x for x in df.columns if x != "id" ]
#
#    ids = [ int2key(x) for x in df["id"] ]
#
#    print(df[df["id"]==0][features].to_numpy())
#    
#    triplets = np.loadtxt(triplet_file)
#    triplets = [ x for x in triplets if int2key(x[0]) in ids and int2key(x[1]) in ids and int2key(x[2]) in ids ]
#    data = np.array([ np.array([ df[df["id"]==x[0]][features].to_numpy(),  df[df["id"]==x[1]][features].to_numpy(), df[df["id"]==x[2]][features].to_numpy() ]) for x in triplets ])
#
#    return data


def make_triplets(array, triplet_file):

    triplets = np.loadtxt(triplet_file)

    data = np.array([ np.array([ array[int(x[0]), :], array[int(x[1]), :], array[int(x[2]), :] ]) for x in triplets ])

    return data


    
def make_0_triplets(data1):

    data0 = np.array([ np.array([ im[0], im[2], im[1] ]) for im in data1 ])
    return data0
    

#resnet_weights_path='./models/resnet50_notop.h5'
def create_model(input_size):

    input_1 = Input(shape=(input_size,))
    input_2 = Input(shape=(input_size,))
    input_3 = Input(shape=(input_size,))

    #difference_1 = subtract([tower_2, tower_1])
    #difference_2 = subtract([tower_3, tower_1])
    #merged = concatenate([difference_1, difference_2], axis=1)
    #merged = Flatten()(merged)

    merged = concatenate([input_1, input_2, input_3], axis=1)

    l1 = Dense(100, activation='relu')(merged)
    l1 = Dropout(0.2)(l1)

    out = Dense(2, activation='softmax')(l1)

    model = Model(inputs=[input_1, input_2, input_3], outputs=[out])

    return model


def not_(x):
    return -x+1


def main(model_name):


    ## Read data
    print("Reading features...")
    train_df = pd.read_csv(features_directory + model_name + "_features.csv", delimiter=',')
    ids = train_df.id.tolist()
    train_df = train_df.drop(["id"], axis=1)

    # Sanity checks: check if ids are from 0 to N
    if [ x for x in range(len(ids)) ] != ids:
        print("ERROR: Indices are not ordered!")
        sys.exit()

    
    ## Normalize features
    print("\nNormalization...")
    scaler = StandardScaler()
    train_array = scaler.fit_transform(train_df)             # scale features so that they have an


    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "./datasets/train_triplets.txt"
    data1 = make_triplets(train_array, triplet_file)
    data0 = make_0_triplets(data1)

    # Sample with permutations of the same triplets
    data1_in = data1
    data0_in = data0
    # Sample without permutations of the same triplets
    data1_in, data1_out, data0_out, data0_in = train_test_split(data1, data0, train_size=0.5)

    n1 = len(data1_in)
    n0 = len(data0_in)
    X = np.concatenate((data1_in, data0_in), axis=0)

    print(np.shape(data1))
    print(np.shape(data0))
    print(np.shape(X))

    ## Make output
    y = np.array( n1*[1.] + n0*[0.] )
    y_2D = np.array(list(map(list, zip(y, not_(y)))))

    print(np.shape(y))
    print(np.shape(y_2D))

    ## Train test split
    print("\nSplitting into training dataset (80%), validation dataset (10%) and test dataset (10%)...")
    X_train, X_vt, y_2D_train, y_2D_vt = train_test_split(X, y_2D, train_size=0.80)
    X_validation, X_test, y_2D_validation, y_2D_test = train_test_split(X_vt, y_2D_vt, train_size=0.5)


    ## Make model
    model = create_model(np.shape(X)[2])
    
    print("Model summary:")
    print(model.summary())


    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['acc', 'AUC'],
    )
   

    print("\nFitting...")
    history = model.fit((X_train[:, 0], X_train[:, 1], X_train[:, 2]), y_2D_train,
                        validation_data=((X_validation[:, 0], X_validation[:, 1], X_validation[:, 2]), y_2D_validation),
                        epochs=30,
                        batch_size=1024)


    ## Prediction on the test dataset
    print("\nPredictions for the test sample...")
    y_pred_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    print(y_pred_proba)
    auc = roc_auc_score(y_2D_test[:, 0], y_pred_proba[:, 0])
    print("ROC AUC: %.2f" %auc)

    cuts = np.logspace(-8, 1, 100)
    y_preds = [ y_pred_proba[:, 0] >= cut for cut in cuts ]
    acc_scores = [ accuracy_score(y_2D_test[:, 0], y_preds[icut]) for icut in range(len(cuts)) ]


    ## Accuracy as a fct of cut
    plt.figure()
    plt.plot(cuts, acc_scores)
    plt.xlabel("Cut value")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.savefig("acc_score.pdf")
    plt.close()


    ## Best accuracy
    best_cut_idx = np.argmax(acc_scores)
    best_cut = cuts[best_cut_idx]
    y_pred = y_preds[best_cut_idx]

    print(y_pred)
    print("Accuracy: %.3f" %(accuracy_score(y_2D_test[:, 0], y_pred)))


    ## Control plots
    # Loss
    plt.figure()
    plot_var("loss", history)
    plt.savefig("loss.pdf")
    plt.close()

    # AUC
    plt.figure()
    plot_var("auc", history)
    plt.savefig("auc.pdf")
    plt.close()

    # Accurarcy
    plt.figure()
    plot_var("acc", history)
    plt.savefig("acc.pdf")
    plt.close()


    ## Load test dataset
    print("\nPredictions for the test dataset...")
    triplet_file = "./datasets/test_triplets.txt"
    X_test = make_triplets(train_array, triplet_file)
    y_pred_test_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    y_pred_test = y_pred_test_proba[:,0] >= best_cut

    np.savetxt("submit.txt", y_pred_test, fmt="%d")


    return


if __name__ == "__main__":

    model_name = "VGG19"

    main(model_name)
