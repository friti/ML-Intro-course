import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import svm, neighbors, neural_network
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import datetime
import pickle
import json
import sys


## Read training data, labels, test data
train_features = pd.read_csv('train_features_preprocessed.csv', delimiter=',')
train_labels   = pd.read_csv('dataset/train_labels.csv'  , delimiter=',')
test_features  = pd.read_csv('test_features_preprocessed.csv', delimiter=',')

# Sort in pid
train_features = train_features.sort_values(['pid'])
pids_features = train_features["pid"].drop_duplicates().to_list()

train_labels   = train_labels[train_labels["pid"].isin(pids_features)].sort_values(['pid'])
test_features  = test_features.sort_values(['pid'])

# Sanity checks
pids_labels   = train_labels["pid"].drop_duplicates().to_list()

if pids_features != pids_labels:
    print("PID ERROR")
    print(pids_features)
    print(pids_labels)
    sys.exit()


# Use pid as index
train_features.set_index("pid", inplace=True)
train_labels.set_index("pid", inplace=True)

## Define labels to predict
labels_to_predict = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
train_labels = train_labels[labels_to_predict]


## Select a subset of training features
features_names_used = [ x for x in train_features.columns if x.endswith("_avg") ]
#features_names_used = [ x for x in train_features.columns if (x.endswith("_avg") or x.endswith("_slope")) ]
#features_names_used = train_features.columns

train_features = train_features[features_names_used]

print("Used features:")
print(features_names_used)
print("Number of used features: %d" %(len(features_names_used)))


## Split train data into train and validation
random_state = None
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, train_size=0.8, random_state=random_state)

models = {}

## Fit model
for label in labels_to_predict:
    print("\n=== %s ===" %label)
    
    # Define model
#    models[label] = neural_network.MLPRegressor(
#        hidden_layer_sizes=(100),
#	alpha=100,                    # L2 regularization
#	activation="relu",
#	solver="adam",
#	learning_rate_init=0.01,
#	learning_rate="constant",
#	max_iter=500
#    )

    models[label] = Lasso(alpha = 0.1, fit_intercept=True)

    # Fit model
    print(datetime.datetime.now())
    models[label].fit(X_train, y_train[label])
    print(datetime.datetime.now())

    # Prediction to evaluate the model
    y_pred = models[label].predict(X_test)
    print("R2 score: %.2f" %(metrics.r2_score(y_test[label], y_pred)))


## Make predictions
test_features = test_features[features_names_used]
df_predictions = pd.DataFrame()
for label in labels_to_predict:
    df_predictions["pid"] = test_features.index
    df_predictions[label] = models[label].predict(test_features)

df_predictions.set_index("pid", inplace=True)

df_predictions.to_csv("subtask3_predictions.csv")
