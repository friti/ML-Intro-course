#####################################   README   ###############################
# We first tried to train a dense NN using the predicted features from a
# ResNet50, VGG16 or VGG19 pre-trained network.
# Howver, no matter what number of hidden layers, number of units, dropout,
# or L2 regularisation, we never achieved a better accuracy than 0.65.
#
# We then tried to use a Siamese network (3 ResNet50 in parallel + some dense
# layers + triplet loss), where we would train the last CNN layers and/or the
# last dense layers, but could never generalize on the validation dataset.
#
# We provide here both the code used to achieve the best accuracy (NN using
# features from pretrained CNN) and the Siamese network code.
#
################################################################################


## Code used for the getting the bets accuracy

import os
import sys
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Activation, concatenate, subtract
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19
from tensorflow.keras.applications import ResNet50, VGG16, VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers



IMAGE_HEIGHT = 342
IMAGE_WIDTH  = 512


def get_id_from_path(path):
    return int(path.split("/")[-1].split(".")[0])


def make_list_of_image_paths(imageDirectory):

    listNonSorted = [ imageDirectory+f for f in os.listdir(imageDirectory) if (os.path.isfile(imageDirectory+f) and f.endswith(".jpg")) ]
    listSorted = sorted(listNonSorted, key=get_id_from_path)

    return listSorted

def read_and_prep_image(image_path, model_name, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    image_raw = load_img(image_path, target_size=(img_height, img_width))
    image_array = np.array([img_to_array(image_raw)])
    if model_name == "ResNet50":
        image = preprocess_input_ResNet50(image_array)
    elif model_name == "VGG16":
        image = preprocess_input_VGG16(image_array)
    elif model_name == "VGG19":
        image = preprocess_input_VGG19(image_array)
    return image


def create_model_to_make_features(model_name, pooling, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, weights="imagenet"):

    input_ = Input(shape=(img_height, img_width, 3,))

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False,
                              pooling=pooling,
                              weights=weights,
                              input_shape=(img_height, img_width, 3))
    elif model_name == "VGG16":
        base_model = VGG16(include_top=False,
                           pooling=pooling,
                           weights=weights,
                           input_shape=(img_height, img_width, 3))
    elif model_name == "VGG19":
        base_model = VGG19(include_top=False,
                           pooling=pooling,
                           weights=weights,
                           input_shape=(img_height, img_width, 3))
    else:
        print("ERROR: Unknown model!")
        sys.exit(1)

    cnn = Model(inputs=base_model.input, outputs=base_model.layers[-1].output, name=model_name)(input_)
    output = Flatten()(cnn)

    model = Model(inputs=[input_], outputs=[output])

    return model



def make_latent_features(modelName, pooling, featuresDirectory):
 
    ## Initialize output CSV file
    if not os.path.exists(featuresDirectory): os.makedirs(featuresDirectory)

    filename = featuresDirectory + modelName + "_features.csv"
    csvfile = open(filename, 'w')
    csvwriter = csv.writer(csvfile) 

    if modelName == "ResNet50":
        if pooling == None:
            nFeatures = 2048*10*10
        else:
            nFeatures = 2048
    elif modelName == "VGG16":
        if pooling == None:
            nFeatures = 512*9*9
        else:
            nFeatures = 512
    elif modelName == "VGG19":
        if pooling == None:
            nFeatures = 512*9*9
        else:
            nFeatures = 512

    featureNames = [ "id" ] + [ "f"+str(x+1) for x in range(nFeatures) ]
    csvwriter.writerow(featureNames) 


    ## Make list of paths to images
    print("Reading images...")
    imageDirectory = "../datasets/food/"
    listOfImagePaths = make_list_of_image_paths(imageDirectory)


    ## Make model
    print("\nCreate %s model..." %modelName)
    model = create_model_to_make_features(modelName, pooling)
    print("Model summary:")
    print(model.summary())


    ## Loop over all images and write features to csv file
    progress = 0
    for idx, imagePath in enumerate(listOfImagePaths):
        # Show progress
        progress_tmp = (100*(idx+1))//len(listOfImagePaths)
        if progress_tmp % 5 == 0 and progress_tmp>progress:
            progress = progress_tmp
            print("%d%% completed" %progress)

        # Load the image
        image = read_and_prep_image(imagePath, modelName)

        # Compute latent features
        features = model.predict(image)
        row = [ get_id_from_path(imagePath) ] + np.ndarray.tolist(features[0])

        # Write features to csv file
        csvwriter.writerow(row)


    ## Close the csv file
    csvfile.close()

    return



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


def make_train_validation_test_triplets_list(triplet_file, no_repetition=False):

    triplets = np.loadtxt(triplet_file)

    if no_repetition:
	train_triplets_file = "./train_triplets_list.txt"
	validation_triplets_file = "./validation_triplets_list.txt"
	test_triplets_file = "./test_triplets_list.txt"

	if os.path.exists(train_triplets_file) and os.path.exists(validation_triplets_file) and os.path.exists(test_triplets_file):
	    triplets_train = np.loadtxt(train_triplets_file)
	    triplets_validation = np.loadtxt(validation_triplets_file)
	    triplets_test = np.loadtxt(test_triplets_file)
	    
	else:
	    train_images = random.sample(range(0, 5000), 3800)

	    triplets_train = [ t for t in triplets if (t[0] in train_images     and t[1] in train_images     and t[2] in train_images)     ]
	    triplets_vt    = [ t for t in triplets if (t[0] not in train_images and t[1] not in train_images and t[2] not in train_images) ]

	    triplets_validation, triplets_test = train_test_split(triplets_vt, train_size=0.5)

	    np.savetxt(train_triplets_file, triplets_train)
	    np.savetxt(validation_triplets_file, triplets_validation)
	    np.savetxt(test_triplets_file, triplets_test)

    else:
	triplets_train, triplets_vt = train_test_split(triplets, train_size=0.8)
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
 

def create_model(input_size, n_units, dropout, regularisation):

    input_1 = Input(shape=(input_size,))
    input_2 = Input(shape=(input_size,))
    input_3 = Input(shape=(input_size,))

    #dropout0 = 0.8
    #n0 = 300
    #l1_1 = Dense(n0, activation='relu')(input_1)
    #l1_1 = Dropout(dropout0)(l1_1)
    #l1_2 = Dense(n0, activation='relu')(input_2)
    #l1_2 = Dropout(dropout0)(l1_2)
    #l1_3 = Dense(n0, activation='relu')(input_3)
    #l1_3 = Dropout(dropout0)(l1_3)
    #merged = concatenate([l1_1, l1_2, l1_3], axis=1)
    #merged = BatchNormalization()(merged)


    merged = concatenate([input_1, input_2, input_3], axis=1)
    l1 = Dense(n_units, activation='relu', kernel_regularizer=regularizers.l2(regularisation))(merged)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(dropout)(l1)

    #l2 = Dense(n_units, activation='relu')(l1)
    #l2 = BatchNormalization()(l2)
    #l2 = Dropout(dropout)(l2)

    out = Dense(2, activation='softmax')(l1)

    model = Model(inputs=[input_1, input_2, input_3], outputs=[out])

    return model


def not_(x):
    return -x+1


def prep_data(model_name, features_directory):

    ## Read data
    print("\nReading features...")
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
    features_array = scaler.fit_transform(train_df)
    #features_array = train_df.to_numpy()

    return features_array


def fit(model_name, params, features_array):

    ## Print parameters
    print("\nRun parameters:")
    for k, v in params.items():
        print("%s: %.3f" %(k, v))


    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "../datasets/train_triplets.txt"


    ## Make train and validation triplets making sure there are no common images in the train and validation triplets
    triplets_train, triplets_validation, triplets_test = make_train_validation_test_triplets_list(triplet_file)

    data_train_1 = make_triplets(features_array, triplets_train)
    data_validation_1 = make_triplets(features_array, triplets_validation)
    data_test_1 = make_triplets(features_array, triplets_test)

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
    model = create_model(np.shape(X_train)[2], params["n_units"], params["dropout"], params["regularisation"])
    
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

    cuts = np.logspace(-2, 1, 100)
    y_preds = [ y_pred_proba[:, 0] >= cut for cut in cuts ]
    acc_scores = [ accuracy_score(y_2D_test[:, 0], y_preds[icut]) for icut in range(len(cuts)) ]


    ## Best accuracy cut
    best_cut_idx = np.argmax(acc_scores)
    best_cut = cuts[best_cut_idx]
    #print("Best cut: %.3f" %best_cut)
    y_pred = y_preds[best_cut_idx]

    if abs(best_cut-0.5)>0.1:
        print("WARNING: Check best cut!")
        print("Best cut: %.3f" %best_cut)

    best_cut = 0.5
    y_pred = y_pred_proba[:, 0] >= best_cut

    accuracy = accuracy_score(y_2D_test[:, 0], y_pred)
    print("Accuracy: %.3f" %accuracy)


    ## Control plots
    # Loss
    for variable in ("loss", "acc"):
        plt.figure()
        plot_var(variable, history)
        plt.savefig(variable + ".pdf")
        plt.close()


    return model


def predict(model, features_array):

    ## Load test dataset
    print("\nPredictions for the test dataset...")
    triplet_file = "../datasets/test_triplets.txt"
    X_test = make_triplets_from_file(features_array, triplet_file)
    y_pred_test_proba = model.predict((X_test[:, 0], X_test[:, 1], X_test[:, 2]))
    y_pred_test = y_pred_test_proba[:, 0] >= best_cut

    np.savetxt("submit.txt", y_pred_test, fmt="%d")




if __name__ == "__main__":

    featuresDirectory = "preprocessed_features/"
    model_name = "VGG19"

    #pooling = "max"
    #make_latent_features(model_name, pooling, featuresDirectory)

    params = {
        "n_epochs"      : 50,
        "batch_size"    : 2500,
        "dropout"       : 0.6,
        "n_units"       : 32,
        "regularisation": 0.0,
    }


    features_array = prep_data(model_name, featuresDirectory)
    model = fit(model_name, params, features_array)
    predict(model, features_array)
