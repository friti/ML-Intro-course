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
# metrics
from sklearn.metrics import roc_auc_score


## Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Activation, concatenate, subtract
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

 
IMAGE_HEIGHT = 240
IMAGE_WIDTH  = 350

def read_and_prep_images(img_path, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    img_names = [ f for f in os.listdir(img_path) if (os.path.isfile(os.path.join(img_path, f)) and f.endswith(".jpg")) ]
    img_names = img_names[:2500]
    imgs = [ load_img(os.path.join(img_path, img_name), target_size=(img_height, img_width)) for img_name in img_names ]
    imgs = np.array([img_to_array(img) for img in imgs])
    imgs = preprocess_input(imgs)
    imgs = { k.split(".")[0]: v for k, v in zip(img_names, imgs) }

    return imgs

    #imgs = { img_name.split(".")[0]: load_img(os.path.join(img_path, img_name), target_size=(img_height, img_width)) for img_name in img_names }
    # Normalize
    #imgs = { k: v/255. for k, v in imgs.items() }

    #img_array = np.array([img_to_array(img) for img in imgs])
    #return output


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
def create_model(img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, resnet_weights="imagenet"):

    input_1 = Input(shape=(img_height, img_width, 3,))
    input_2 = Input(shape=(img_height, img_width, 3,))
    input_3 = Input(shape=(img_height, img_width, 3,))


    res_net = ResNet50(include_top=False,
                       pooling='avg',
                       weights=resnet_weights,
                       input_shape=(img_height, img_width, 3))
    for layer in res_net.layers:
        layer.trainable = False

    tower_1 = Model(inputs=res_net.input, outputs=res_net.layers[-1].output, name='resnet50_1')(input_1)
    tower_2 = Model(inputs=res_net.input, outputs=res_net.layers[-1].output, name='resnet50_2')(input_2)
    tower_3 = Model(inputs=res_net.input, outputs=res_net.layers[-1].output, name='resnet50_3')(input_3)
    #tower_1.trainable = False
    #tower_2.trainable = False
    #tower_3.trainable = False

    #tower_1 = ResNet50(include_top=False,
    #                   pooling='avg',
    #                   weights=resnet_weights) \
    #          (input_1)
    #tower_1 = MaxPooling2D((1, 9), strides=(1, 1), padding='same')(tower_1)

    #tower_2 = ResNet50(include_top=False,
    #                   pooling='avg',
    #                   weights=resnet_weights) \
    #          (input_2)
    ##tower_2 = MaxPooling2D((1, 9), strides=(1, 1), padding='same')(tower_2)
    #tower_2.trainable = False

    #tower_3 = ResNet50(include_top=False,
    #                   pooling='avg',
    #                   weights=resnet_weights) \
    #          (input_3)
    ##tower_3 = MaxPooling2D((1, 6), strides=(1, 1), padding='same')(tower_3)
    #tower_3.trainable = False

    difference_1 = subtract([tower_2, tower_1])
    difference_2 = subtract([tower_3, tower_1])

    #merged = concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = concatenate([difference_1, difference_2], axis=1)
    merged = Flatten()(merged)
    merged = Dropout(0.2)(merged)

    out = Dense(20, activation='relu')(merged)
    #out1 = Dropout(0.2)(out1)
    #out2 = Dense(20, activation='relu')(out2)
    out = Dense(2, activation='softmax')(out)

    #model = Model(input_shape, out)
    model = Model(inputs=[input_1, input_2, input_3], outputs=[out])

    return model


def not_(x):
    return -x+1


def main():


    ## Read data
    print("Reading images...")
    img_path = "./datasets/food/"
    images = read_and_prep_images(img_path)

    #keys = list(images.keys())
    #print(keys)
    #print(images[keys[0]])

    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "./datasets/train_triplets.txt"
    data1 = make_triplets(images, triplet_file)
    data0 = make_0_triplets(data1)
    X = np.concatenate((data1, data0), axis=0)

    print(np.shape(data1))
    print(np.shape(data0))
    print(np.shape(X))

    ## Make output
    y = np.array( len(data1) * [1.] + len(data0) * [0.] )
    y_2D = np.array(list(map(list, zip(y, not_(y)))))

    print(np.shape(y))
    print(np.shape(y_2D))

    ## Train test split
    print("\nSplitting into training dataset (80%), validation dataset (10%) and test dataset (10%)...")
    X_train, X_vt, y_2D_train, y_2D_vt = train_test_split(X, y_2D, train_size=0.80)
    X_validation, X_test, y_2D_validation, y_2D_test = train_test_split(X_vt, y_2D_vt, train_size=0.5)


    ## Make model
    model = create_model()
    
    print("Model summary:")
    print(model.summary())


    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['acc', 'AUC'],
    )
   

    print("\nFitting...")
    history = model.fit((X_train[:,0], X_train[:,1], X_train[:,2]), y_2D_train,
                        validation_data=((X_validation[:,0], X_validation[:,1], X_validation[:,2]), y_2D_validation),
                        epochs=20,
                        batch_size=250)


    ## Prediction on the test dataset
    print("\nPredictions for the test sample...")
    y_pred_proba = model.predict((X_test[:,0], X_test[:,1], X_test[:,2]))
    print(y_pred_proba)
    auc = roc_auc_score(y_2D_test[:,0], y_pred_proba[:,0])
    print("ROC AUC: %.2f" %auc)

    #cuts = np.logspace(-8, 1, 100)
    #y_preds = [ y_pred_proba[:,0]>=cut for cut in cuts ]
    #f1_scores = [ f1_score(y_test, y_preds[icut]) for icut in range(len(cuts)) ]

#
#    ## F1 score as a fct of cut
#    plt.figure()
#    plt.plot(cuts, f1_scores)
#    plt.xlabel("Cut value")
#    plt.ylabel("F1 score")
#    plt.xscale("log")
#    plt.savefig("f1_score.pdf")
#    plt.close()
#
#
#    ## Best F1 score
#    best_cut_idx = np.argmax(f1_scores)
#    best_cut = cuts[best_cut_idx]
#    y_pred = y_preds[best_cut_idx]
#
#    print(y_pred)
#    print("F1 score: %.3f" %(f1_score(y_test, y_pred)))
#
#
#    def plot_var(variable, history):
#        plt.title(variable)
#        plt.plot(history.history[variable][2:], label='train')
#        plt.plot(history.history['val_'+variable][2:], label='validation')
#        plt.legend()
#        plt.xlabel("Number of epochs")
#        plt.ylabel(variable)
#        plt.grid(True)
#
#
#
#    ## Loss
#    plt.figure()
#    plot_var("loss", history)
#    plt.savefig("loss.pdf")
#    plt.close()
#
#
#    ## AUC
#    plt.figure()
#    plot_var("auc", history)
#    plt.savefig("auc.pdf")
#    plt.close()
#
#
#    ## Load test dataset
#    test_df = pd.read_csv('dataset/test.csv', delimiter=',')
#    print("\nPredictions for the test dataset...")
#
#    test_df_encoded = test_df.copy()
#    columns_encoded = one_hot_encoding(test_df_encoded, amino_acids)
#    X_test = test_df_encoded[columns_encoded]
#
#    y_pred_test_proba = model.predict(X_test)
#    y_pred_test = y_pred_test_proba[:,0]>=best_cut
#
#    np.savetxt("submit.txt", y_pred_test, fmt="%d")


    return


if __name__ == "__main__":
    main()
