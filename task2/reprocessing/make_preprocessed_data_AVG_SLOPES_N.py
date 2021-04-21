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


## Read training data, labels, test data
train_features_original = pd.read_csv('dataset/train_features.csv', delimiter=',')
train_labels_original   = pd.read_csv('dataset/train_labels.csv'  , delimiter=',')
test_features_original  = pd.read_csv('dataset/test_features.csv' , delimiter=',')

# sort train data and labels values by pid, so they are the same (because after with the manipulation of data things get nasty)
train_features_original = train_features_original.sort_values(['pid','Time'])
train_labels_original   = train_labels_original.sort_values(['pid'])
test_features_original  = test_features_original.sort_values(['pid','Time'])

# Use pid as index
train_labels_original.set_index("pid", inplace=True)

print(train_features_original.head())
print(train_labels_original.head())

# Make list of PIDs
pids_original = train_features_original["pid"].drop_duplicates().to_list()
Npatients_original = len(pids_original)
print("Number of patients: %d" %Npatients_original)
print("Dataset length: %d" %(len(train_features_original)))

# Reducing training for speeding up tests
Npatients_lite = Npatients_original//100
pids_lite = pids_original[:Npatients_lite]

train_features_lite = train_features_original[train_features_original["pid"].isin(pids_lite)]
print("Lite number of patients: %d" %Npatients_lite)
print("Lite dataset length: %d" %(len(train_features_lite)))

train_labels_lite = train_labels_original[train_labels_original.index.isin(pids_lite)]

# Decide here if to use the lite dataset or the whole dataset

#train_features = train_features_lite.copy()
#train_labels   = train_labels_lite.copy()
#pids = pids_lite.copy()
#Npatients = Npatients_lite

train_features = train_features_original.copy()
train_labels   = train_labels_original.copy()
pids = pids_original.copy()
Npatients = Npatients_original

# Make list of feature names
feature_names = [ x for x in train_features.columns  if x not in ("pid", "Time") ]
print("Features:")
print(feature_names)


# Replace the series of 12 measurements by their average and slope
print(datetime.datetime.now())

def make_linear_fit(x, y):
    x = x.where(y.notna()).dropna()
    y = y.dropna()
    if len(x) < 2:
        return np.nan
    else:
        return np.polyfit(x, y, 1)[0]
    
    
counts = train_features[["pid"]+feature_names].groupby(["pid"]).count().add_suffix("_n")
avgs = train_features[["pid"]+feature_names].groupby(["pid"]).mean().add_suffix("_avg")
#train_features_preprocessed = counts.copy()
train_features_preprocessed = pd.concat([counts, avgs], axis=1)
train_features_preprocessed["pid"] = train_features["pid"].copy()

train_features_preprocessed["time_list"] = train_features.groupby(["pid"]).Time.apply(list)
for feature_name in feature_names:
    if feature_name == "Age":
        train_features_preprocessed[feature_name + "_slope"] = Npatients * [0.]
    else:
        train_features_preprocessed[feature_name + "_list"] = train_features.groupby(["pid"])[feature_name].apply(list)
        train_features_preprocessed[feature_name + "_slope"] = train_features_preprocessed.apply(lambda row: make_linear_fit(pd.Series(row["time_list"]), pd.Series(row[feature_name + "_list"])), axis=1)
        # Delete the _list columns
        train_features_preprocessed.drop(feature_name + "_list", axis=1, inplace=True)

    print("%s finished at %s" %(feature_name, datetime.datetime.now()))
    
# Delete the time_list columns
train_features_preprocessed.drop("time_list", axis=1, inplace=True)
    
print(datetime.datetime.now())



# Replace NaNs of a column by the average of the column
feature_averages = {}
for feature_name in feature_names:
    avgs = train_features_preprocessed[feature_name + "_avg"].replace(np.nan, 0)
    slopes = train_features_preprocessed[feature_name + "_slope"].replace(np.nan, 0)
    
    avg_avg = np.average(avgs, weights=train_features_preprocessed[feature_name + "_n"])
    slope_avg = np.average(slopes, weights=train_features_preprocessed[feature_name + "_n"])

    train_features_preprocessed[feature_name + "_avg"].replace(np.nan, avg_avg, inplace=True)
    train_features_preprocessed[feature_name + "_slope"].replace(np.nan, slope_avg, inplace=True)



# Make features to use in training
def std_scaler(array):
    mean = np.mean(array)
    std = np.std(array, ddof=1)
    if std != 0:
        return (array-mean)/std
    else:
        return array

# Add features
for feature_name in feature_names:
    # Std scaling
    train_features_preprocessed[feature_name + "_avg"] = std_scaler(train_features_preprocessed[feature_name + "_avg"])
    train_features_preprocessed[feature_name + "_slope"] = std_scaler(train_features_preprocessed[feature_name + "_slope"])
    
print(train_features_preprocessed.head())
print(len(train_features_preprocessed))


## Save to csv file
train_features_preprocessed.to_csv("preprocessed_data.csv")
