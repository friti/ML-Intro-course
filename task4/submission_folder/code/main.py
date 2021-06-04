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
from sklearn.utils import shuffle
# metrics
from sklearn.metrics import roc_auc_score, accuracy_score

## Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Activation, concatenate, subtract
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers
from make_latent_features import make_latent_features
features_directory = "../preprocessed_features/"

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

'''def make_train_validation_test_triplets_list(triplet_file):

    triplets = np.loadtxt(triplet_file)
    print(len(triplets))

    train_triplets_file = "./train_triplets_list.txt"
    validation_triplets_file = "./validation_triplets_list.txt"
    test_triplets_file = "./test_triplets_list.txt"

    if os.path.exists(train_triplets_file) and os.path.exists(validation_triplets_file) and os.path.exists(test_triplets_file):
        triplets_train = np.loadtxt(train_triplets_file)
        triplets_validation = np.loadtxt(validation_triplets_file)
        triplets_test = np.loadtxt(test_triplets_file)
        
    else:
        # Construct train, validation and test samples such that there are no repetitions of images
        train_images = list(range(0, 3800))

        triplets_train = [ t for t in triplets if (t[0] in train_images     and t[1] in train_images     and t[2] in train_images)     ]
        triplets_vt    = [ t for t in triplets if (t[0] not in train_images and t[1] not in train_images and t[2] not in train_images) ]

        triplets_validation, triplets_test = train_test_split(triplets_vt, train_size=0.5)

        np.savetxt(train_triplets_file, triplets_train)
        np.savetxt(validation_triplets_file, triplets_validation)
        np.savetxt(test_triplets_file, triplets_test)

    print("Train dataset size:      %d" %(len(triplets_train)))
    print("Validation dataset size: %d" %(len(triplets_validation)))
    print("Test dataset size:       %d" %(len(triplets_test)))


    return triplets_train, triplets_validation, triplets_test'''
def make_train_validation_test_triplets_list(triplet_file):

    triplets = np.loadtxt(triplet_file)

    # sample part of the triplets
    n_triplets = len(triplets)
    #n_triplets = 10000
    #triplets = triplets[sample_without_replacement(n_population=n_triplets, n_samples=8000)]

    triplets_train, triplets_vt = train_test_split(triplets[:n_triplets], train_size=0.8)
    triplets_validation, triplets_test = train_test_split(triplets_vt, train_size=0.5)
    print("Train dataset size:      %d" %(len(triplets_train)))
    print("Validation dataset size: %d" %(len(triplets_validation)))
    print("Test dataset size:       %d" %(len(triplets_test)))


    return triplets_train, triplets_validation, triplets_test
    

def make_triplets(array, triplets_list):
    data = np.array([ np.array([ array[int(x[0]), :], array[int(x[1]), :], array[int(x[2]), :] ]) for x in triplets_list ])
    return data


def make_triplets_from_file(array, triplet_file):

    triplets = np.loadtxt(triplet_file)
    data = np.array([ np.array([ array[int(x[0]), :], array[int(x[1]), :], array[int(x[2]), :] ]) for x in triplets ])

    return data
 

    
def make_0_triplets(data1):

    data0 = np.array([ np.array([ im[0], im[2], im[1] ]) for im in data1 ])
    return data0
    

def create_model(input_size, n_units, dropout):

    input_1 = Input(shape=(input_size,))
    input_2 = Input(shape=(input_size,))
    input_3 = Input(shape=(input_size,))

    #difference_1 = subtract([input_2, input_1])
    #difference_2 = subtract([input_3, input_1])
    #merged = concatenate([difference_1, difference_2], axis=1)

    merged = concatenate([input_1, input_2, input_3], axis=1)

    #l1 = Dense(n_units, activation='relu',kernel_regularizer=regularizers.l2(1e-2),bias_regularizer=regularizers.l2(0.01))(merged)
    l1 = Dense(n_units, activation='relu',kernel_regularizer=regularizers.l2(1e-2))(merged)
#    l1 = Dense(n_units, activation='relu')(merged)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(dropout)(l1)

    l2 = Dense(128, activation='relu')(l1)
    l2 = BatchNormalization()(l2)
    l2 = Dropout(0.5)(l2)

    l3 = Dense(64, activation='relu')(l2)
    #l3 = BatchNormalization()(l3)
    l3 = Dropout(0.4)(l3)

    out = Dense(2, activation='softmax')(l3)

    model = Model(inputs=[input_1, input_2, input_3], outputs=[out])

    return model


def not_(x):
    return -x+1


def main(model_name,params):


    ## Read data
    print("Reading features...")
    train_df = pd.read_csv(features_directory + model_name + "_features.csv", delimiter=',')
    ids = train_df.id.tolist()
    train_df = train_df.drop(["id"], axis=1)

    # For features PCA only! Keep only first features
    #n_features_kept = 200
    #features_kept = [ "f"+str(i+1) for i in range(n_features_kept) ]
    #train_df = train_df[features_kept]

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
    triplet_file = "../datasets/train_triplets.txt"

    ## Old triplet code
    ##data1 = make_triplets(train_array, triplet_file)
    ##data0 = make_0_triplets(data1)

    ## Sample with permutations of the same triplets
    ##data1_in = data1
    ##data0_in = data0
    ## Sample without permutations of the same triplets
    #data1_in, data1_out, data0_out, data0_in = train_test_split(data1, data0, train_size=0.5)

    #n1 = len(data1_in)
    #n0 = len(data0_in)
    #X = np.concatenate((data1_in, data0_in), axis=0)

    #print(np.shape(data1))
    #print(np.shape(data0))
    #print(np.shape(X))

    ## Make output
    #y = np.array( n1*[1.] + n0*[0.] )
    #y_2D = np.array(list(map(list, zip(y, not_(y)))))

    #print(np.shape(y))
    #print(np.shape(y_2D))

    ### Train test split
    #print("\nSplitting into training dataset (80%), validation dataset (10%) and test dataset (10%)...")
    #X_train, X_vt, y_2D_train, y_2D_vt = train_test_split(X, y_2D, train_size=0.80)
    #X_validation, X_test, y_2D_validation, y_2D_test = train_test_split(X_vt, y_2D_vt, train_size=0.5)


    ## Make train and validation triplets making sure there are no common images in the train and validation triplets
    triplets_train, triplets_validation, triplets_test = make_train_validation_test_triplets_list(triplet_file)

    data_train_1 = make_triplets(train_array, triplets_train)
    data_validation_1 = make_triplets(train_array, triplets_validation)
    data_test_1 = make_triplets(train_array, triplets_test)

    data_train_0 = make_0_triplets(data_train_1)
    data_validation_0 = make_0_triplets(data_validation_1)
    data_test_0 = make_0_triplets(data_test_1)

    data_train_1_in, data_train_1_out, data_train_0_out, data_train_0_in = train_test_split(data_train_1, data_train_0, train_size=0.5)
    data_validation_1_in, data_validation_1_out, data_validation_0_out, data_validation_0_in = train_test_split(data_validation_1, data_validation_0, train_size=0.5)
    data_test_1_in, data_test_1_out, data_test_0_out, data_test_0_in = train_test_split(data_test_1, data_test_0, train_size=0.5)

    n1 = len(data_train_1_in)
    n0 = len(data_train_0_in)
    X_train = np.concatenate((data_train_1_in, data_train_0_in), axis=0)
    y_train = np.array( n1*[1.] + n0*[0.] )
    y_2D_train = np.array(list(map(list, zip(y_train, not_(y_train)))))

    n1 = len(data_validation_1_in)
    n0 = len(data_validation_0_in)
    X_validation = np.concatenate((data_validation_1_in, data_validation_0_in), axis=0)
    y_validation = np.array( n1*[1.] + n0*[0.] )
    y_2D_validation = np.array(list(map(list, zip(y_validation, not_(y_validation)))))

    n1 = len(data_test_1_in)
    n0 = len(data_test_0_in)
    X_test = np.concatenate((data_test_1_in, data_test_0_in), axis=0)
    y_test = np.array( n1*[1.] + n0*[0.] )
    y_2D_test = np.array(list(map(list, zip(y_test, not_(y_test)))))


    ## Shuffle the datasets
    X_train, y_2D_train           = shuffle(X_train, y_2D_train)
    X_validation, y_2D_validation = shuffle(X_validation, y_2D_validation)


    ## Make model
    model = create_model(np.shape(X_train)[2], params["n_units"], params["dropout"])
        
    print("Model summary:")
    print(model.summary())


    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['acc'],
    )
   

    print("\nFitting...")
    history = model.fit((X_train[:, 0], X_train[:, 1], X_train[:, 2]), y_2D_train,
                        validation_data=((X_validation[:, 0], X_validation[:, 1], X_validation[:, 2]), y_2D_validation),
                        epochs=params["n_epochs"],
                        batch_size=params["batch_size"])


    ## Prediction on the test dataset
    print("\nPredictions for the test sample...")
    y_pred_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    auc = roc_auc_score(y_2D_test[:, 0], y_pred_proba[:, 0])
    print("ROC AUC: %.2f" %auc)

    #cuts = np.logspace(-2, 1, 100)
    #y_preds = [ y_pred_proba[:, 0] >= cut for cut in cuts ]
    #acc_scores = [ accuracy_score(y_2D_test[:, 0], y_preds[icut]) for icut in range(len(cuts)) ]
    # 

    #### Accuracy as a fct of cut
    #plt.figure()
    #plt.plot(cuts, acc_scores)
    #plt.xlabel("Cut value")
    #plt.ylabel("Accuracy")
    #plt.xscale("log")
    #plt.savefig("acc_score.pdf")
    #plt.close()


    ### Best accuracy cut
    #best_cut_idx = np.argmax(acc_scores)
    #best_cut = cuts[best_cut_idx]
    #print("Best cut: %.3f" %best_cut)
    #y_pred = y_preds[best_cut_idx]

    best_cut = 0.5
    y_pred = y_pred_proba[:, 0] >= best_cut

    print("Accuracy: %.3f" %(accuracy_score(y_2D_test[:, 0], y_pred)))


    ## Control plots
    # Loss
    '''for variable in ("loss", "acc"):
        plt.figure()
        plot_var(variable, history)
        plt.savefig(variable + ".pdf")
        plt.close()
    '''
    #sys.exit()

    ## Load test dataset
    print("\nPredictions for the test dataset...")
    triplet_file = "../datasets/test_triplets.txt"
    X_test = make_triplets_from_file(train_array, triplet_file)
    y_pred_test_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    y_pred_test = y_pred_test_proba[:, 0] >= best_cut

    np.savetxt("submit.txt", y_pred_test, fmt="%d")
    

    return


if __name__ == "__main__":

    model_name = "VGG19"

    params = {	"n_epochs": 70,
		"batch_size": 1024,
		"dropout": 0.7,
		"n_units": 256,
	      }
    
    # Uncomment this to preprocess the images through a pretrained CNN

    #pooling = "max"
    #make_latent_features(model_name, pooling)
    main(model_name,params)
    