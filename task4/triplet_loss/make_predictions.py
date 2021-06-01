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
from keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, concatenate, subtract
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19
from tensorflow.keras.applications import ResNet50, VGG16, VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



# GLOBAL DEFINES
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 342


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

    return [ [t[0], t[2], t[1]] for t in triplets ]


def make_triplets_from_file(array, triplet_file):

    triplets = np.loadtxt(triplet_file)
    data = np.array([ np.array([ array[int(x[0]), :], array[int(x[1]), :], array[int(x[2]), :] ]) for x in triplets ])

    return data
 

def not_(x):
    return -x+1


def make_0_1_triplets(triplet_file):
    ## Make train and validation triplets making sure there are no common images in the train and validation triplets
    triplets_train_1, triplets_validation_1, triplets_test_1 = make_train_validation_test_triplets_list(triplet_file)
    triplets_train_0      = make_triplets_0(triplets_train_1)
    triplets_validation_0 = make_triplets_0(triplets_validation_1)
    triplets_test_0       = make_triplets_0(triplets_test_1)
    
    triplets_train_1_in, triplets_train_1_out, triplets_train_0_out, triplets_train_0_in                     = train_test_split(triplets_train_1, triplets_train_0, train_size=0.5)
    triplets_validation_1_in, triplets_validation_1_out, triplets_validation_0_out, triplets_validation_0_in = train_test_split(triplets_validation_1, triplets_validation_0, train_size=0.5)
    triplets_test_1_in, triplets_test_1_out, triplets_test_0_out, triplets_test_0_in                         = train_test_split(triplets_test_1, triplets_test_0, train_size=0.5)

    return triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in


def make_triplets_from_0_1_triplets(triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in):

    triplets_train = np.concatenate((triplets_train_1_in, triplets_train_0_in), axis=0)
    triplets_validation = np.concatenate((triplets_validation_1_in, triplets_validation_0_in), axis=0)
    triplets_test = np.concatenate((triplets_test_1_in, triplets_test_0_in), axis=0)

    return triplets_train, triplets_validation, triplets_test


def make_labels(triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in):

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

    return y_train, y_validation, y_test, y_2D_train, y_2D_validation, y_2D_test


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0])+K.square(y_pred[:,2,0])) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def create_model():

    n_units = 100

    # Initialize a ResNet50_ImageNet Model
    resnet_input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    resnet_model = ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)

    # New Layers over ResNet50
    net = resnet_model.output
    #net = Flatten(name='flatten')(net)
    net = GlobalAveragePooling2D(name='gap')(net)
    #net = Dropout(0.5)(net)
    net = Dense(n_units, activation='relu',name='t_emb_1')(net)
    net = Lambda(lambda x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)

    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_pos')
    input_negative = Input(shape=input_shape, name='input_neg')

    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    tertiary_dist = Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])

    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    # Setting up optimizer designed for variable learning rate

    # Variable Learning Rate per Layers
    lr_mult_dict = {}
    last_layer = ''
    for layer in resnet_model.layers:
        # comment this out to refine earlier layers
        # layer.trainable = False  
        # print layer.name
        lr_mult_dict[layer.name] = 1
        # last_layer = layer.name
    lr_mult_dict['t_emb_1'] = 100

    #base_lr = 0.0001
    #momentum = 0.9
    #v_optimizer = LR_SGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, multipliers = lr_mult_dict)

    #model.compile(optimizer=v_optimizer, loss=triplet_loss, metrics=[accuracy])

    return model


def make_list_of_image_paths(imageDirectory):
    return [ imageDirectory+("%05d.jpg" %x) for x in range(10000) ]


def read_and_prep_image(image_path, modelName, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    image_raw = load_img(image_path, target_size=(img_height, img_width))
    image_array = np.array([img_to_array(image_raw)])
    if modelName == "ResNet50":
        image = preprocess_input_ResNet50(image_array)
    elif modelName == "VGG16":
        image = preprocess_input_VGG16(image_array)
    elif modelName == "VGG19":
        image = preprocess_input_VGG19(image_array)
    return image



## loads an image and preprocesses
#def read_image(loc):
#    image = cv2.imread(loc)
#    image = cv2.resize(t_image, (IMAGE_HEIGHT,IMAGE_WIDTH))
#    image = image.astype("float32")
#    image = keras.applications.resnet50.preprocess_input(image, data_format='channels_last')
#
#    return t_image
#
## loads a set of images from a text index file   
#def read_image_list(flist, start, length):
#
#    with open(flist) as f:
#        content = f.readlines() 
#    content = [x.strip().split()[0] for x in content] 
#
#    datalen = length
#    if (datalen < 0):
#        datalen = len(content)
#
#    if (start + datalen > len(content)):
#        datalen = len(content) - start
# 
#    imgset = np.zeros((datalen, IMAGE_HEIGHT, IMAGE_WIDTH, T_G_NUMCHANNELS))
#
#    for i in range(start, start+datalen):
#        if ((i-start) < len(content)):
#            imgset[i-start] = t_read_image(content[i])
#
#    return imgset


def main():

    n_epochs = 2
    chunksize = 50

    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "../datasets/train_triplets.txt"

    triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in = make_0_1_triplets(triplet_file)
    y_train, y_validation, y_test, y_2D_train, y_2D_validation, y_2D_test = make_labels(triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in)
    triplets_train, triplets_validation, triplets_test = make_triplets_from_0_1_triplets(triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in)

    ## Shuffle the datasets
    triplets_train, y_train, y_2D_train                = shuffle(triplets_train, y_train, y_2D_train)
    triplets_validation, y_validation, y_2D_validation = shuffle(triplets_validation, y_validation, y_2D_validation)

    ## Find number of passes through the dataset
    n_triplets_train = len(triplets_train)
    n_chunks = int(n_triplets_train // chunksize)
    chunksize_validation = int(len(triplets_validation) // n_chunks)

    ## Make model
    model = create_model()
    
    print("Model summary:")
    print(model.summary())


    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['acc'],
    )
   

    ## Make list of paths to images
    print("Reading images...")
    imageDirectory = "../datasets/food/"
    listOfImagePaths = make_list_of_image_paths(imageDirectory)


    ## Loop
    print("\nFitting...")
    histories = []
    for i_chunks in range(n_chunks):

        print("Chunk number %d" %i_chunks)

        idx_train_1 = i_chunks*chunksize
        idx_train_2 = (i_chunks+1)*chunksize
        idx_val_1 = i_chunks*chunksize_validation
        idx_val_2 = (i_chunks+1)*chunksize_validation

        
        # Load the images
        modelName = "ResNet50"
        anchors_train   = [ read_and_prep_image(listOfImagePaths[int(t[0])], modelName) for t in triplets_train[idx_train_1:idx_train_2] ]
        print("1")
        positives_train = [ read_and_prep_image(listOfImagePaths[int(t[1])], modelName) for t in triplets_train[idx_train_1:idx_train_2] ]
        print("2")
        negatives_train = [ read_and_prep_image(listOfImagePaths[int(t[2])], modelName) for t in triplets_train[idx_train_1:idx_train_2] ]
        print("3")

        anchors_validation   = [ read_and_prep_image(listOfImagePaths[int(t[0])], modelName) for t in triplets_validation[idx_val_1:idx_val_2] ]
        print("4")
        positives_validation = [ read_and_prep_image(listOfImagePaths[int(t[1])], modelName) for t in triplets_validation[idx_val_1:idx_val_2] ]
        print("5")
        negatives_validation = [ read_and_prep_image(listOfImagePaths[int(t[2])], modelName) for t in triplets_validation[idx_val_1:idx_val_2] ]
        print("6")

        print("Images are read")

        histories.append(model.fit([anchors_train, positives_train, negatives_train]), y_2D_train[idx_train_1:idx_train_2],
                            validation_data=([anchors_validation, positives_validation, negatives_validation], y_2D_validation[idx_train_1:idx_train_2]),
                            epochs=n_epochs,
                            batch_size=10)

        #print(triplets_train[idx_train_1:idx_train_2])
        #print(triplets_validation[idx_val_1:idx_val_2])


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


if __name__ == "__main__":
    main()
