import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
import tensorflow as tf
from pathlib import Path
from tensorflow.python.keras.models import load_model, save_model
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet



#IMAGE_WIDTH = 512
#IMAGE_HEIGHT = 342
IMAGE_WIDTH = 350
IMAGE_HEIGHT = 240
target_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )



class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.acc_tracker = metrics.Mean(name="acc")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            acc = self._compute_acc(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def _compute_acc(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        #acc = tf.Transform.mean(ap_distance < an_distance)
        acc = tf.math.reduce_mean(tf.cast(ap_distance < an_distance, dtype=tf.float32))
        return acc

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]



def create_model():

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable


    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return embedding, siamese_network


def create_embedding():

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    return embedding


def make_train_validation_test_triplets_list(triplet_file):

    triplets = np.loadtxt(triplet_file)

    # sample part of the triplets
    n_triplets = len(triplets)
    triplets = triplets[sample_without_replacement(n_population=n_triplets, n_samples=2000)]

    train_triplets_file = "./train_triplets_list.txt"
    validation_triplets_file = "./validation_triplets_list.txt"
    test_triplets_file = "./test_triplets_list.txt"

    if os.path.exists(train_triplets_file) and os.path.exists(validation_triplets_file) and os.path.exists(test_triplets_file):
        triplets_train = np.loadtxt(train_triplets_file)
        triplets_validation = np.loadtxt(validation_triplets_file)
        triplets_test = np.loadtxt(test_triplets_file)
        
    else:
        train_images = random.sample(range(0, 5000), 2500)  #list(range(0, 3800))

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
 

#def make_0_1_triplets(triplet_file):
#    ## Make train and validation triplets making sure there are no common images in the train and validation triplets
#    triplets_train_1, triplets_validation_1, triplets_test_1 = make_train_validation_test_triplets_list(triplet_file)
#    triplets_train_0      = make_0_triplets(triplets_train_1)
#    triplets_validation_0 = make_0_triplets(triplets_validation_1)
#    triplets_test_0       = make_0_triplets(triplets_test_1)
#    
#    triplets_train_1_in, triplets_train_1_out, triplets_train_0_out, triplets_train_0_in                     = train_test_split(triplets_train_1, triplets_train_0, train_size=0.5)
#    triplets_validation_1_in, triplets_validation_1_out, triplets_validation_0_out, triplets_validation_0_in = train_test_split(triplets_validation_1, triplets_validation_0, train_size=0.5)
#    triplets_test_1_in, triplets_test_1_out, triplets_test_0_out, triplets_test_0_in                         = train_test_split(triplets_test_1, triplets_test_0, train_size=0.5)
#
#    return triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in
#
#
#def make_triplets_from_0_1_triplets(triplets_train_1_in, triplets_train_0_in, triplets_validation_1_in, triplets_validation_0_in, triplets_test_1_in, triplets_test_0_in):
#
#    triplets_train = np.concatenate((triplets_train_1_in, triplets_train_0_in), axis=0)
#    triplets_validation = np.concatenate((triplets_validation_1_in, triplets_validation_0_in), axis=0)
#    triplets_test = np.concatenate((triplets_test_1_in, triplets_test_0_in), axis=0)
#
#    return triplets_train, triplets_validation, triplets_test


def make_list_of_image_paths(imageDirectory):
    return [ imageDirectory+("%05d.jpg" %x) for x in range(10000) ]


def load_triplets(triplets_train, triplets_validation, triplets_test):

    batch_size = 64

    imageDirectory = "../datasets/food/"
    listOfImagePaths = make_list_of_image_paths(imageDirectory)

    anchor_images_train   = [ listOfImagePaths[int(t[0])] for t in triplets_train ]
    positive_images_train = [ listOfImagePaths[int(t[1])] for t in triplets_train ]
    negative_images_train = [ listOfImagePaths[int(t[2])] for t in triplets_train ]

    anchor_images_val   = [ listOfImagePaths[int(t[0])] for t in triplets_validation ]
    positive_images_val = [ listOfImagePaths[int(t[1])] for t in triplets_validation ]
    negative_images_val = [ listOfImagePaths[int(t[2])] for t in triplets_validation ]

    anchor_images_test   = [ listOfImagePaths[int(t[0])] for t in triplets_test ]
    positive_images_test = [ listOfImagePaths[int(t[1])] for t in triplets_test ]
    negative_images_test = [ listOfImagePaths[int(t[2])] for t in triplets_test ]

    anchor_dataset_train   = tf.data.Dataset.from_tensor_slices(anchor_images_train)
    positive_dataset_train = tf.data.Dataset.from_tensor_slices(positive_images_train)
    negative_dataset_train = tf.data.Dataset.from_tensor_slices(negative_images_train)

    anchor_dataset_val   = tf.data.Dataset.from_tensor_slices(anchor_images_val)
    positive_dataset_val = tf.data.Dataset.from_tensor_slices(positive_images_val)
    negative_dataset_val = tf.data.Dataset.from_tensor_slices(negative_images_val)

    anchor_dataset_test   = tf.data.Dataset.from_tensor_slices(anchor_images_test)
    positive_dataset_test = tf.data.Dataset.from_tensor_slices(positive_images_test)
    negative_dataset_test = tf.data.Dataset.from_tensor_slices(negative_images_test)

    dataset_train = tf.data.Dataset.zip((anchor_dataset_train, positive_dataset_train, negative_dataset_train)).shuffle(buffer_size=1024)
    dataset_val   = tf.data.Dataset.zip((anchor_dataset_val  , positive_dataset_val  , negative_dataset_val  )).shuffle(buffer_size=1024)
    dataset_test  = tf.data.Dataset.zip((anchor_dataset_test , positive_dataset_test , negative_dataset_test )).shuffle(buffer_size=1024)
    dataset_train = dataset_train.map(preprocess_triplets)
    dataset_val   = dataset_val.map(preprocess_triplets)
    dataset_test  = dataset_test.map(preprocess_triplets)

    dataset_train = dataset_train.batch(batch_size, drop_remainder=False)
    dataset_val   = dataset_val.batch(batch_size, drop_remainder=False)
    dataset_test  = dataset_test.batch(batch_size, drop_remainder=False)

    return dataset_train, dataset_val, dataset_test


def distance(f1, f2):
    """Compute distance between arrays of features."""

    return np.sum((np.sum([f1, -f2], axis=0))**2, axis=1)


def make_predictions(features_a, features_b, features_c):
    """Compute predictions"""

    return distance(features_a, features_b) < distance(features_a, features_c)


def main_train():

    
    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "../datasets/train_triplets.txt"
    triplets_train, triplets_validation, triplets_test = make_train_validation_test_triplets_list(triplet_file)
    dataset_train, dataset_val, dataset_test = load_triplets(triplets_train, triplets_validation, triplets_test)


    embedding, siamese_network = create_model()
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))

    checkpoint_filepath = './checkpoints/{epoch:02d}-{val_loss:.2f}.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_loss',
        save_best_only=False,
        save_freq="epoch",
    )
    siamese_model.fit(dataset_train, epochs=3, validation_data=dataset_val, callbacks=[model_checkpoint_callback])

    embedding.save_weights("test.h5")



def main_predict():

    ## Make train data
    print("\nMaking datasets...")
    triplet_file = "../datasets/train_triplets.txt"
    triplets_train, triplets_validation, triplets_test = make_train_validation_test_triplets_list(triplet_file)
    dataset_train, dataset_val, dataset_test = load_triplets(triplets_train, triplets_validation, triplets_test)

    embedding, siamese_network = create_model()
    siamese_network.load_weights("./checkpoints/03-0.66.h5")

    a = dataset_test.map(lambda a, b, c: a)
    b = dataset_test.map(lambda a, b, c: b)
    c = dataset_test.map(lambda a, b, c: c)

    features_a = embedding.predict(a)
    features_b = embedding.predict(b)
    features_c = embedding.predict(c)

    predictions = make_predictions(features_a, features_b, features_c)
    print(predictions)
    print(np.mean(predictions==1))


if __name__ == "__main__":

    main_train()
    #main_predict()

