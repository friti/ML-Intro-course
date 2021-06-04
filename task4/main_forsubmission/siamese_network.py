import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
from pathlib import Path
from tensorflow.python.keras.models import load_model, save_model
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
target_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

ROOT_DIRECTORY = "./"
IMAGE_DIRECTORY = ROOT_DIRECTORY + "task4_data/food/food/"
TRIPLETS_DIRECTORY = ROOT_DIRECTORY + "task4_data/"
CHECKPOINTS_DIRECTORY = ROOT_DIRECTORY + "task4_data/checkpoints/"


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to match the target size
    image = tf.image.resize(image, target_shape)
    image = resnet.preprocess_input(image)
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

augment_image = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomZoom(0.2),
  layers.experimental.preprocessing.RandomContrast(0.3),
])


def augment_triplets(anchor, positive, negative):
    """
    Augment a triplet.
    """

    return (
        augment_image(anchor),
        augment_image(positive),
        augment_image(negative),
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
            ap_distance, an_distance = self._compute_distances(data)
            loss = self._compute_loss(ap_distance, an_distance)
            acc = self._compute_acc(ap_distance, an_distance)

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
        ap_distance, an_distance = self._compute_distances(data)
        loss = self._compute_loss(ap_distance, an_distance)
        acc = self._compute_acc(ap_distance, an_distance)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def _compute_distances(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)
        return ap_distance, an_distance

    def _compute_loss(self, ap_distance, an_distance):
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def _compute_acc(self, ap_distance, an_distance):
        # Cmomputing the accuracy
        acc = tf.math.reduce_mean(tf.cast(ap_distance < an_distance, dtype=tf.float32))
        return acc

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


def create_embedding():

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), pooling="max", include_top=False
    )

    dense1 = layers.Dropout(0.3)(base_cnn.output)
    output = layers.Dense(128)(dense1)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block3_1_conv":
            trainable = True
        layer.trainable = trainable

    return embedding


def create_model():

    embedding = create_embedding()


    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return embedding, siamese_network


def make_train_validation_test_triplets_list(triplet_file, random_seed=None):

    random.seed(random_seed)

    triplets = np.loadtxt(triplet_file)

    # sample part of the triplets
    n_triplets = len(triplets)
    triplets = triplets[sample_without_replacement(n_population=n_triplets, n_samples=40000)]

    train_triplets_file = "./train_triplets_list.txt"
    validation_triplets_file = "./validation_triplets_list.txt"
    test_triplets_file = "./test_triplets_list.txt"

    if os.path.exists(train_triplets_file) and os.path.exists(validation_triplets_file) and os.path.exists(test_triplets_file):
        triplets_train = np.loadtxt(train_triplets_file)
        triplets_validation = np.loadtxt(validation_triplets_file)
        triplets_test = np.loadtxt(test_triplets_file)
        
    else:
        train_images = random.sample(range(0, 5000), 3600)  #list(range(0, 3800))

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


def make_list_of_image_paths(image_directory):
    return [ image_directory+("%05d.jpg" %x) for x in range(10000) ]


def make_anchor_positive_negative_datasets(list_of_image_paths, triplets):

    anchor_images   = [ list_of_image_paths[int(t[0])] for t in triplets ]
    positive_images = [ list_of_image_paths[int(t[1])] for t in triplets ]
    negative_images = [ list_of_image_paths[int(t[2])] for t in triplets ]

    anchor_dataset   = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

    return anchor_dataset, positive_dataset, negative_dataset


def preprocess_dataset(list_of_image_paths, triplets, batch_size=16, shuffle=False, augment=False, random_seed=None):

    # Make anchor, positive, negativ images datasets
    anchor_dataset, positive_dataset, negative_dataset = make_anchor_positive_negative_datasets(list_of_image_paths, triplets)
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
     
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024, seed=random_seed)
      
    # Preprocess
    dataset = dataset.map(preprocess_triplets)

   # Batch size
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Data augmentation
    if augment:
        dataset = dataset.map(augment_triplets)

   
    return dataset


def preprocess_dataset_test(list_of_image_paths, triplets, batch_size=16, random_seed=None):

    # Make anchor, positive, negativ images datasets
    anchor_dataset, positive_dataset, negative_dataset = make_anchor_positive_negative_datasets(list_of_image_paths, triplets)
    
    # Preprocess
    anchor_dataset = anchor_dataset.map(preprocess_image)
    positive_dataset = positive_dataset.map(preprocess_image)
    negative_dataset = negative_dataset.map(preprocess_image)
 
    # Batch size
    anchor_dataset = anchor_dataset.batch(batch_size, drop_remainder=False)
    positive_dataset = positive_dataset.batch(batch_size, drop_remainder=False)
    negative_dataset = negative_dataset.batch(batch_size, drop_remainder=False)

    return anchor_dataset, positive_dataset, negative_dataset


def load_triplets(triplets_train, triplets_validation, triplets_test, image_directory=IMAGE_DIRECTORY, random_seed=None):

    batch_size = 32

    list_of_image_paths = make_list_of_image_paths(image_directory)

    dataset_train = preprocess_dataset(list_of_image_paths, triplets_train     , batch_size=batch_size, shuffle=True, augment=True , random_seed=random_seed)
    dataset_val   = preprocess_dataset(list_of_image_paths, triplets_validation, batch_size=batch_size, shuffle=True, augment=False, random_seed=random_seed)
    dataset_test  = preprocess_dataset(list_of_image_paths, triplets_test      , batch_size=batch_size, shuffle=True, augment=False, random_seed=random_seed)

    return dataset_train, dataset_val, dataset_test


def load_test_dataset(triplets, image_directory=IMAGE_DIRECTORY, random_seed=None):

    batch_size = 32

    list_of_image_paths = make_list_of_image_paths(image_directory)
    anchor_dataset, positive_dataset, negative_dataset = preprocess_dataset_test(list_of_image_paths, triplets, batch_size=batch_size, random_seed=random_seed)

    return anchor_dataset, positive_dataset, negative_dataset


def distance(f1, f2):
    """Compute distance between arrays of features."""

    return np.sum((np.sum([f1, -f2], axis=0))**2, axis=1)


def make_predictions(features_a, features_b, features_c):
    """Compute predictions"""

    d_ab = distance(features_a, features_b)
    d_ac = distance(features_a, features_c)
    pred = d_ab < d_ac
    return pred


def main_train(random_seed=1234):

    
    ## Make train data
    print("\nMaking datasets...")
    triplet_file = TRIPLETS_DIRECTORY + "train_triplets.txt"
    triplets_train, triplets_validation, triplets_test = make_train_validation_test_triplets_list(triplet_file, random_seed=random_seed)
    dataset_train, dataset_val, dataset_test = load_triplets(triplets_train, triplets_validation, triplets_test,random_seed=random_seed)


    embedding, siamese_network = create_model()
    
    print("\nSiamese summary:")
    print(siamese_network.summary())

    print("\nEmbedding summary:")
    print(embedding.summary())

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam())

    
    if not os.path.exists(CHECKPOINTS_DIRECTORY): os.makedirs(CHECKPOINTS_DIRECTORY)
    checkpoint_filepath = CHECKPOINTS_DIRECTORY + '{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=False,
        save_freq="epoch",
    )
    siamese_model.fit(dataset_train, epochs=10, validation_data=dataset_val, callbacks=[model_checkpoint_callback])

    embedding.save_weights("test_emb.h5")
    siamese_network.save_weights("test_sia.h5")

    return embedding, siamese_network


def main_predict(embedding=None, random_seed=1234):

    ## Make train data
    print("\nMaking datasets...")
    triplet_file = TRIPLETS_DIRECTORY + "train_triplets.txt"
    triplets_train, triplets_validation, triplets_test = make_train_validation_test_triplets_list(triplet_file, random_seed=random_seed)
    anchor_dataset, positive_dataset, negative_dataset = load_test_dataset(triplets_validation, random_seed=random_seed)

    if embedding is None:
        embedding, siamese_network = create_model()
        siamese_network.load_weights(CHECKPOINTS_DIRECTORY + "03-0.66.h5")
    else:
        embedding = embedding


    features_a = embedding.predict(anchor_dataset)
    features_b = embedding.predict(positive_dataset)
    features_c = embedding.predict(negative_dataset)

    predictions = make_predictions(features_a, features_b, features_c)
    print(predictions)
    print(np.mean(predictions==1))



random_seed = None
embedding, siamese_network = main_train(random_seed=random_seed)
main_predict(embedding=embedding, random_seed=random_seed)

