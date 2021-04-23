import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import json
import sys
import os

from sklearn import metrics
from sklearn import svm, neighbors, neural_network, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


def make_linear_fit(x, y):
    """
    Make linear fit
    x and y are 2 pd.Series with possible NaNs
    """

    x = x.where(y.notna()).dropna()
    y = y.dropna()
    if len(x) < 2:
        return np.nan
    else:
        return np.polyfit(x, y, 1)[0]


def std_scaler(array, array_n):
    """Standard Scaler"""

    mean = np.average(array.dropna(), weights=array_n.where(array.notna()).dropna())
    std = np.std(array, ddof=1)
    if std != 0:
        return (array-mean)/std, mean, std
    else:
        return array, mean, std


def scale(array, mean, std):
    """Apply Standard Scaler"""

    if std != 0:
        return (array-mean)/std
    else:
        return array


        
def make_preprocessed_features(file_name, repfile_name, is_train=True, use_slopes=False):
    """
    Make preprocessed features to be used for training and prediction.
    Saves new features into a file.
    
    Normalisation for train dataset is saved into a json file and the same mean and std dev
    are used for normalising the test features.

    NaNs are replaced by the average of the columns (0. after normalisation).
    """

    # Read training data, labels, test data
    dataframe = pd.read_csv(file_name, delimiter=',')


    # Sort by PID and time for convenience
    dataframe = dataframe.sort_values(['pid', 'Time'])


    # Make list of PIDs
    pids = dataframe["pid"].drop_duplicates().to_list()
    Npatients = len(pids)
    print("Number of patients: %d" %Npatients)
    print("Dataset length: %d" %(len(dataframe)))


    # Make list of feature names
    feature_names = [ x for x in dataframe.columns  if x not in ("pid", "Age", "Time") ]
    print("Features:")
    print(feature_names)


    # Replace the series of 12 measurements by their average and slope
    print(datetime.datetime.now())

    ages = dataframe[["pid", "Age"]].groupby(["pid"]).mean()
    counts = dataframe[["pid"]+feature_names].groupby(["pid"]).count().add_suffix("_n")/12  # Divide by 12 to get number in between 0 and 1
    avgs = dataframe[["pid"]+feature_names].groupby(["pid"]).mean().add_suffix("_avg")
    mins = dataframe[["pid"]+feature_names].groupby(["pid"]).min().add_suffix("_min")
    maxs = dataframe[["pid"]+feature_names].groupby(["pid"]).max().add_suffix("_max")
    df_preprocessed = pd.concat([ages, counts, avgs, mins, maxs], axis=1)
    for feature_name in feature_names:
        df_preprocessed[feature_name + "_diff"] = df_preprocessed[feature_name + "_max"] - df_preprocessed[feature_name + "_min"]

    feature_suffixes = [ "n", "avg", "min", "max", "diff" ]

    if use_slopes:
        feature_suffixes = feature_suffixes + ["slope"]
        df_preprocessed["time_list"] = dataframe.groupby(["pid"]).Time.apply(list)
        for feature_name in feature_names:
            if feature_name == "Age":
                df_preprocessed[feature_name + "_slope"] = Npatients * [0.]
            else:
                df_preprocessed[feature_name + "_list"] = dataframe.groupby(["pid"])[feature_name].apply(list)
                df_preprocessed[feature_name + "_slope"] = df_preprocessed.apply(lambda row: make_linear_fit(pd.Series(row["time_list"]), pd.Series(row[feature_name + "_list"])), axis=1)
                # Delete the _list columns
                df_preprocessed.drop(feature_name + "_list", axis=1, inplace=True)

            print("%s finished at %s" %(feature_name, datetime.datetime.now()))
            
        # Delete the time_list columns
        df_preprocessed.drop("time_list", axis=1, inplace=True)
        
    print(datetime.datetime.now())


    # Normalize features
    if is_train:
        normalisation = {}
        for feature_name in feature_names:
            for suffix in feature_suffixes:
                if suffix == "n": continue   # No scaling for number of measurements
                feature_name_full = feature_name + "_" + suffix
                # Std scaling
                df_preprocessed[feature_name_full], avg, std = std_scaler(df_preprocessed[feature_name_full], df_preprocessed[feature_name + "_n"])
                print("%s mean: %.3f   std: %.3f" %(feature_name_full, avg, std))
                normalisation[feature_name_full] = {"mean": avg, "std": std}

        # Age
        feature_name_full = "Age"
        df_preprocessed[feature_name_full], avg, std = std_scaler(df_preprocessed[feature_name_full], df_preprocessed[feature_name + "_n"])
        print("%s mean: %.3f   std: %.3f" %(feature_name_full, avg, std))
        normalisation[feature_name_full] = {"mean": avg, "std": std}

        with open('normalisation.json', 'w') as f:
           json.dump(normalisation, f)

    else:
        with open('normalisation.json') as f:
            normalisation = json.load(f)
        for feature_name in feature_names:
            for suffix in feature_suffixes:
                if suffix == "n": continue   # No scaling for number of measurements
                feature_name_full = feature_name + "_" + suffix
                df_preprocessed[feature_name_full] = scale(df_preprocessed[feature_name_full], normalisation[feature_name_full]["mean"], normalisation[feature_name_full]["std"])

        # Age
        feature_name_full = "Age"
        df_preprocessed[feature_name_full] = scale(df_preprocessed[feature_name_full], normalisation[feature_name_full]["mean"], normalisation[feature_name_full]["std"])


    # Replace NaNs of a column by the average of the column
    if is_train:
        imputation = {}
        for feature_name in feature_names:
            for suffix in feature_suffixes:
                if suffix == "n": continue   # No imputation for number of measurements (no NaNs)
                feature_name_full = feature_name + "_" + suffix
                feature = df_preprocessed[feature_name_full].replace(np.nan, 0)
                feature_avg = np.average(feature, weights=df_preprocessed[feature_name + "_n"])
                print("%s avg over not NaN  : %.3f" %(feature_name_full, feature_avg))
                imputation[feature_name_full] = feature_avg
                df_preprocessed[feature_name_full].replace(np.nan, feature_avg, inplace=True)

        with open('imputation.json', 'w') as f:
           json.dump(imputation, f)

    else:
        with open('imputation.json') as f:
            imputation = json.load(f)

        for feature_name in feature_names:
            for suffix in feature_suffixes:
                if suffix == "n": continue   # No imputation for number of measurements (no NaNs)
                feature_name_full = feature_name + "_" + suffix
                feature_avg = imputation[feature_name_full]
                df_preprocessed[feature_name_full].replace(np.nan, feature_avg, inplace=True)


    print("")
    print("=======================")
    print("=== FINAL DATAFRAME ===")
    print("=======================")
    print("")
    print(df_preprocessed.head())

    print("")
    print("")


    # Save to csv file
    df_preprocessed.to_csv(repfile_name)


def make_subtask1(train_features_file, train_labels_file, test_features_file):
    """
    Read preprocessed features, train a model, make prediction on the test features and saves
    predictions into a file.
    """

    # Read training data, labels, test data
    train_features = pd.read_csv(train_features_file, delimiter=',')
    train_labels   = pd.read_csv(train_labels_file, delimiter=',')
    test_features = pd.read_csv(test_features_file, delimiter=',')

    # Sort by pid the training features and labels
    train_features = train_features.sort_values(['pid'])
    train_labels = train_labels.sort_values(['pid'])

    # Also do that for test (for convenience when looking at it)
    test_features  = test_features.sort_values(['pid'])

    # Sanity checks
    pids_features = train_features["pid"].drop_duplicates().to_list()
    pids_labels   = train_labels["pid"].drop_duplicates().to_list()
    if pids_features != pids_labels:
        print("PID ERROR")
        print(len(pids_features))
        print(len(pids_labels))
        sys.exit()


    # Use pid as index
    train_features.set_index("pid", inplace=True)
    train_labels.set_index("pid", inplace=True)

    # Define labels to predict
    labels_to_predict = ["LABEL_BaseExcess","LABEL_Fibrinogen","LABEL_AST","LABEL_Alkalinephos","LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2","LABEL_Bilirubin_direct","LABEL_EtCO2"]
    train_labels = train_labels[labels_to_predict]


    # Dictionary to save output for all labels to predict
    models = {}
    features_names_used = {}

    # To compute the avg over all labels
    score = []

    # Fit model
    for label in labels_to_predict[:]:
        print("\n=== %s ===" %label)

        # Split train data into train and validation
        random_state = None
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels[label], train_size=0.8, random_state=random_state, stratify=train_labels[label])

        # Features to be used for training
        features_names_used[label] = ["Age"] + [ x for x in train_features.columns if (x.endswith("_n") or x.endswith("_avg")) ]
        X_train_used = X_train[features_names_used[label]]
        X_test_used = X_test[features_names_used[label]]

        print("Used features:")
        print("Number of used features: %d" %(len(features_names_used[label])))

        # Define model
        models[label] = linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)

        # Fit model
        models[label].fit(X_train_used, y_train)

        # Prediction to evaluate the model
        y_pred = models[label].predict_proba(X_test_used)[:,1]

        # AUC
        auc = metrics.roc_auc_score(y_test, y_pred)
        print("ROC AUC: %.2f" %auc)
        score.append(auc)

        # Confusion matrix
        matrix = plot_confusion_matrix(models[label], X_test_used, y_test)
        plt.title('Confusion matrix for '+label)
        if not os.path.exists("conf_mat/"): os.makedirs("conf_mat/")
        plt.savefig('conf_mat/'+label+'.png')

    print("\nscore")
    print(np.mean(score))

    # Make predictions and save file
    df_predictions = pd.DataFrame()
    df_predictions["pid"] = test_features["pid"]

    for label in labels_to_predict:
        df = test_features[features_names_used[label]]
        df_predictions[label] = models[label].predict_proba(df)[:,1]

    df_predictions.set_index("pid", inplace=True)
    df_predictions.to_csv("subtask1.csv")


def make_subtask2(train_features_file, train_labels_file, test_features_file):
    """
    Read preprocessed features, train a model, make prediction on the test features and saves
    predictions into a file.
    """

    # Read training data, labels, test data
    train_features = pd.read_csv(train_features_file, delimiter=',')
    train_labels   = pd.read_csv(train_labels_file, delimiter=',')
    test_features = pd.read_csv(test_features_file, delimiter=',')

    # Sort by pid the training features and labels
    train_features = train_features.sort_values(['pid'])
    train_labels = train_labels.sort_values(['pid'])

    # Also do that for test (for convenience when looking at it)
    test_features  = test_features.sort_values(['pid'])

    # Sanity checks
    pids_features = train_features["pid"].drop_duplicates().to_list()
    pids_labels   = train_labels["pid"].drop_duplicates().to_list()
    if pids_features != pids_labels:
        print("PID ERROR")
        print(len(pids_features))
        print(len(pids_labels))
        sys.exit()

    # Use pid as index
    train_features.set_index("pid", inplace=True)
    train_labels.set_index("pid", inplace=True)

    # Define labels to predict
    labels_to_predict = ["LABEL_Sepsis"]
    train_labels = train_labels[labels_to_predict]


    # Dictionary to save output for all labels to predict
    models = {}
    features_names_used = {}

    # To compute the avg over all labels
    score = []

    # Fit model
    for label in labels_to_predict[:]:
        print("\n=== %s ===" %label)

        # Split train data into train and validation
        random_state = None
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels[label], train_size=0.8, random_state=random_state, stratify=train_labels[label])

        # Features to be used for training
        features_names_used[label] = ["Age"] + [ x for x in train_features.columns if (x.endswith("_n") or x.endswith("_avg")) ]
        X_train_used = X_train[features_names_used[label]]
        X_test_used = X_test[features_names_used[label]]

        print("Used features:")
        print("Number of used features: %d" %(len(features_names_used[label])))

        # Define model
        models[label] = linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)

        # Fit model
        models[label].fit(X_train_used, y_train)

        # Prediction to evaluate the model
        y_pred = models[label].predict_proba(X_test_used)[:,1]

        # AUC
        auc = metrics.roc_auc_score(y_test, y_pred)
        print("ROC AUC: %.2f" %auc)
        score.append(auc)

        # Confusion matrix
        matrix = plot_confusion_matrix(models[label], X_test_used, y_test)
        plt.title('Confusion matrix for '+label)
        if not os.path.exists("conf_mat/"): os.makedirs("conf_mat/")
        plt.savefig('conf_mat/'+label+'.png')

    print("\nscore")
    print(np.mean(score))


    # Make predictions
    df_predictions = pd.DataFrame()
    df_predictions["pid"] = test_features["pid"]


    for label in labels_to_predict:
        df = test_features[features_names_used[label]]
        df_predictions[label] = models[label].predict_proba(df)[:,1]

    df_predictions.set_index("pid", inplace=True)
    df_predictions.to_csv("subtask2.csv")


def make_subtask3(train_features_file, train_labels_file, test_features_file):
    """
    Read preprocessed features, train a model, make prediction on the test features and saves
    predictions into a file.
    """

    # Read training data, labels, test data
    train_features = pd.read_csv(train_features_file, delimiter=',')
    train_labels   = pd.read_csv(train_labels_file, delimiter=',')
    test_features = pd.read_csv(test_features_file, delimiter=',')

    # Sort by pid the training features and labels
    train_features = train_features.sort_values(['pid'])
    train_labels = train_labels.sort_values(['pid'])

    # Also do that for test (for convenience when looking at it)
    test_features  = test_features.sort_values(['pid'])

    # Sanity checks
    pids_features = train_features["pid"].drop_duplicates().to_list()
    pids_labels   = train_labels["pid"].drop_duplicates().to_list()
    if pids_features != pids_labels:
        print("PID ERROR")
        print(len(pids_features))
        print(len(pids_labels))
        sys.exit()

    # Use pid as index
    train_features.set_index("pid", inplace=True)
    train_labels.set_index("pid", inplace=True)

    # Define labels to predict
    labels_to_predict = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
    train_labels = train_labels[labels_to_predict]

        
    # Dictionary to save output for all labels to predict
    models = {}
    features_names_used = {}

    # To compute the avg over all labels
    score = []

    # Fit model
    for label in labels_to_predict[:]:
        print("\n=== %s ===" %label)

        # Split train data into train and validation
        random_state = None
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels[label], train_size=0.8, random_state=random_state)

        # Features to be used for training
        features_names_used[label] = ["Age"] + [ x for x in train_features.columns if (x.endswith("_n") or x.endswith("_avg")) ]
        X_train_used = X_train[features_names_used[label]]
        X_test_used = X_test[features_names_used[label]]

        print("Used features:")
        print("Number of used features: %d" %(len(features_names_used[label])))

        # Define model
        models[label] = linear_model.Lasso(alpha = 0.1, fit_intercept=True)

        # Fit model
        models[label].fit(X_train_used, y_train)

        # Prediction to evaluate the model
        y_pred = models[label].predict(X_test_used)
        r2 = metrics.r2_score(y_test, y_pred)
        print("R2 score: %.2f" %r2)

        score.append(r2)

    print("\nscore")
    print(0.5 + 0.5*np.mean(score))


    # Make predictions
    df_predictions = pd.DataFrame()
    df_predictions["pid"] = test_features["pid"]

    for label in labels_to_predict:
        df = test_features[features_names_used[label]]
        df_predictions[label] = models[label].predict(df)

    df_predictions.set_index("pid", inplace=True)
    df_predictions.to_csv("subtask3.csv")


def make_sumbission_file():
    sub1 = pd.read_csv('subtask1.csv', delimiter=',')
    sub2 = pd.read_csv('subtask2.csv', delimiter=',')
    sub3 = pd.read_csv('subtask3.csv', delimiter=',')

    sub1.set_index("pid", inplace=True)
    sub2.set_index("pid", inplace=True)
    sub3.set_index("pid", inplace=True)

    result = pd.concat([sub1,sub2,sub3], axis = 1)

    filename = 'submit'
    compression_options = dict(method='zip', archive_name=f'{filename}.csv')
    result.to_csv(f'{filename}.zip', compression=compression_options, index=True, float_format='%.3f')



if __name__ == "__main__":

    ## Make preprocessed features
    print("=====================")
    print("=== Preprocessing ===")
    print("=====================")
    use_slopes = False

    train_features_file = '../dataset/train_features.csv'
    test_features_file = '../dataset/test_features.csv'
    train_labels_file = '../dataset/train_labels.csv'

    outpath = "preprocessed_files/"
    if not os.path.exists(outpath): os.makedirs(outpath)

    train_repfeatures_file = outpath + "train_features.csv"
    test_repfeatures_file = outpath + "test_features.csv"

    print("Train features preprocessing...")
    make_preprocessed_features(train_features_file, train_repfeatures_file, is_train=True, use_slopes=use_slopes)
    print("\nTest features preprocessing...")
    make_preprocessed_features(test_features_file, test_repfeatures_file, is_train=False, use_slopes=use_slopes)

    print("")
    print("==========================================")
    print("=== Training and predictions subtask 1 ===")
    print("==========================================")
    make_subtask1(train_repfeatures_file, train_labels_file, test_repfeatures_file)

    print("")
    print("==========================================")
    print("=== Training and predictions subtask 2 ===")
    print("==========================================")
    make_subtask2(train_repfeatures_file, train_labels_file, test_repfeatures_file)

    print("")
    print("==========================================")
    print("=== Training and predictions subtask 3 ===")
    print("==========================================")
    make_subtask3(train_repfeatures_file, train_labels_file, test_repfeatures_file)

    make_sumbission_file()

