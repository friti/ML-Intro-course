## OS
import os
import sys
import csv


## Maths
import numpy as np


## Graphics
import matplotlib.pyplot as plt


## DataFrames
import pandas as pd


## Sklearn
# preprocessing
from sklearn.model_selection import train_test_split
# metrics
from sklearn.metrics import roc_auc_score


## Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Activation, concatenate, subtract
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet50, VGG16, VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

 
IMAGE_HEIGHT = 240
IMAGE_WIDTH  = 350

def get_id_from_path(path):
    return int(path.split("/")[-1].split(".")[0])

def make_list_of_image_paths(imageDirectory):

    listNonSorted = [ imageDirectory+f for f in os.listdir(imageDirectory) if (os.path.isfile(imageDirectory+f) and f.endswith(".jpg")) ]
    listSorted = sorted(listNonSorted, key=get_id_from_path)

    return listSorted

def read_and_prep_image(image_path, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    image_raw = load_img(image_path, target_size=(img_height, img_width))
    image_array = np.array([img_to_array(image_raw)])
    image = preprocess_input(image_array)
    return image


def int2key(x):
    return "%05d" %x

def make_triplets(images, triplet_file):
    
    triplets = np.loadtxt(triplet_file)
    triplets = [ x for x in triplets if int2key(x[0]) in images.keys() and int2key(x[1]) in images.keys() and int2key(x[2]) in images.keys() ]
    data = np.array([ np.array([images[int2key(x[0])], images[int2key(x[1])], images[int2key(x[2])]]) for x in triplets ])

    return data

    
def make_0_triplets(data1):

    data0 = np.array([ np.array([ im[0], im[2], im[1] ]) for im in data1 ])
    return data0
    

#resnet_weights_path='./models/resnet50_notop.h5'
def create_model(img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, weights="imagenet", modelName="ResNet50"):

    input_ = Input(shape=(img_height, img_width, 3,))


    if modelName == "ResNet50":
        base_model = ResNet50(include_top=False,
                              pooling='avg',
                              weights=weights,
                              input_shape=(img_height, img_width, 3))
    elif modelName == "VGG16":
        base_model = VGG16(include_top=False,
                           pooling='avg',
                           weights=weights,
                           input_shape=(img_height, img_width, 3))
    elif modelName == "VGG19":
        base_model = VGG19(include_top=False,
                           pooling='avg',
                           weights=weights,
                           input_shape=(img_height, img_width, 3))
    else:
        print("ERROR: Unknown model!")
        sys.exit(1)

    cnn = Model(inputs=base_model.input, outputs=base_model.layers[-1].output, name=modelName)(input_)
    output = Flatten()(cnn)

    model = Model(inputs=[input_], outputs=[output])

    return model


def not_(x):
    return -x+1


def main(modelName):

    ## Initialize output CSV file
    featuresDirectory = "preprocessed_features/"
    if not os.path.exists(featuresDirectory): os.makedirs(featuresDirectory)

    filename = featuresDirectory + modelName + "_features.csv"
    csvfile = open(filename, 'w')
    csvwriter = csv.writer(csvfile) 

    if modelName == "ResNet50": nFeatures = 2048
    elif modelName == "VGG16": nFeatures = 512
    elif modelName == "VGG19": nFeatures = 512
    featureNames = [ "id" ] + [ "f"+str(x+1) for x in range(nFeatures) ]
    csvwriter.writerow(featureNames) 


    ## Make list of paths to images
    print("Reading images...")
    imageDirectory = "./datasets/food/"
    listOfImagePaths = make_list_of_image_paths(imageDirectory)


    ## Make model
    print("\nCreate %s model..." %modelName)
    model = create_model(modelName=modelName)
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
        image = read_and_prep_image(imagePath)

        # Compute latent features
        features = model.predict(image)
        row = [ get_id_from_path(imagePath) ] + np.ndarray.tolist(features[0])

        # Write features to csv file
        csvwriter.writerow(row)


    ## Close the csv file
    csvfile.close()

    return


if __name__ == "__main__":

    modelName = "ResNet50"

    #main(modelName=modelName)
