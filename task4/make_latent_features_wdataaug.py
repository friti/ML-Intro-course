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
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19
from tensorflow.keras.applications import ResNet50, VGG16, VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
 
IMAGE_HEIGHT = 342
IMAGE_WIDTH  = 512

def get_id_from_path(path):
    return int(path.split("/")[-1].split(".")[0])

def make_list_of_image_paths(imageDirectory):

    listNonSorted = [ imageDirectory+f for f in os.listdir(imageDirectory) if (os.path.isfile(imageDirectory+f) and f.endswith(".jpg")) ]
    listSorted = sorted(listNonSorted, key=get_id_from_path)

    return listSorted

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


def create_model(modelName, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, weights="imagenet"):

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
    featuresDirectory = "preprocessed_features_max_size/"
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
    imageDirectory = "dataset/food/"
    listOfImagePaths = make_list_of_image_paths(imageDirectory)


    ## Make model
    print("\nCreate %s model..." %modelName)
    model = create_model(modelName)
    print("Model summary:")
    print(model.summary())


    ## Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')    
    ## Loop over all images and write features to csv file
    progress = 0
    print(len(listOfImagePaths))
    for idx, imagePath in enumerate(listOfImagePaths):
        # Show progress
        progress_tmp = (100*(idx+1))//len(listOfImagePaths)
        if progress_tmp % 5 == 0 and progress_tmp>progress:
            progress = progress_tmp
            print("%d%% completed" %progress)

        # Load the image
        image = read_and_prep_image(imagePath, modelName)
        features = model.predict(image)
        row = [ get_id_from_path(imagePath) ] + np.ndarray.tolist(features[0])

        # Write features to csv file
        csvwriter.writerow(row)

        i = 1
        for batch in datagen.flow(image, batch_size=1):
            features = model.predict(batch)
            row = [ get_id_from_path(imagePath) + 10000*i ] + np.ndarray.tolist(features[0])
            csvwriter.writerow(row)
            i += 1
            #print(batch)
            
            if i > 5:
                break
            #print(i)
            #sys.exit()
        # Compute latent features


    ## Close the csv file
    csvfile.close()

    return


if __name__ == "__main__":

    modelName = "VGG19"

    main(modelName=modelName)
