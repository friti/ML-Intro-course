import pandas as pd
import numpy as np
import datetime
import pickle
import json



def make_linear_fit(x, y):
    x = x.where(y.notna()).dropna()
    y = y.dropna()
    if len(x) < 2:
        return np.nan
    else:
        return np.polyfit(x, y, 1)[0]


def std_scaler(array):
    mean = np.mean(array)
    std = np.std(array, ddof=1)
    if std != 0:
        return (array-mean)/std, mean, std
    else:
        return array, mean, std


def scale(array, mean, std):
    if std != 0:
        return (array-mean)/std
    else:
        return array


        
def main(df, name, is_train=True, use_slopes=False):

    ## Sort by PID and time for convenience
    df = df.sort_values(['pid', 'Time'])

    print(df.head())


    ## Make list of PIDs
    pids = df["pid"].drop_duplicates().to_list()
    Npatients = len(pids)
    print("Number of patients: %d" %Npatients)
    print("Dataset length: %d" %(len(df)))


    ## Reducing training for speeding up tests
    #Npatients = Npatients//10
    #pids = pids[:Npatients]
    #df = df[df["pid"].isin(pids)]
    #print("Lite number of patients: %d" %Npatients)
    #print("Lite dataset length: %d" %(len(df)))


    ## Make list of feature names
    feature_names = [ x for x in df.columns  if x not in ("pid", "Age", "Time") ]
    print("Features:")
    print(feature_names)


    ## Replace the series of 12 measurements by their average and slope
    print(datetime.datetime.now())

        
    ages = df[["pid", "Age"]].groupby(["pid"]).mean()
    counts = df[["pid"]+feature_names].groupby(["pid"]).count().add_suffix("_n")
    avgs = df[["pid"]+feature_names].groupby(["pid"]).mean().add_suffix("_avg")
    mins = df[["pid"]+feature_names].groupby(["pid"]).min().add_suffix("_min")
    maxs = df[["pid"]+feature_names].groupby(["pid"]).max().add_suffix("_max")
    df_preprocessed = pd.concat([ages, counts, avgs, mins, maxs], axis=1)
    for feature_name in feature_names:
        df_preprocessed[feature_name + "_diff"] = df_preprocessed[feature_name + "_max"] - df_preprocessed[feature_name + "_min"]

    feature_suffixes = [ "n", "avg", "min", "max", "diff" ]

    if use_slopes:
        feature_suffixes = feature_suffixes + ["slope"]
        df_preprocessed["time_list"] = df.groupby(["pid"]).Time.apply(list)
        for feature_name in feature_names:
            if feature_name == "Age":
                df_preprocessed[feature_name + "_slope"] = Npatients * [0.]
            else:
                df_preprocessed[feature_name + "_list"] = df.groupby(["pid"])[feature_name].apply(list)
                df_preprocessed[feature_name + "_slope"] = df_preprocessed.apply(lambda row: make_linear_fit(pd.Series(row["time_list"]), pd.Series(row[feature_name + "_list"])), axis=1)
                # Delete the _list columns
                df_preprocessed.drop(feature_name + "_list", axis=1, inplace=True)

            print("%s finished at %s" %(feature_name, datetime.datetime.now()))
            
        # Delete the time_list columns
        df_preprocessed.drop("time_list", axis=1, inplace=True)
        
    print(datetime.datetime.now())



    ## Replace NaNs of a column by the average of the column
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


    ## Normalize features
    if is_train:
        normalisation = {}
        for feature_name in feature_names:
            for suffix in feature_suffixes:
                if suffix == "n": continue   # No scaling for number of measurements
                feature_name_full = feature_name + "_" + suffix
                # Std scaling
                df_preprocessed[feature_name_full], avg, std = std_scaler(df_preprocessed[feature_name_full])
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

    print(df_preprocessed.head())
    print(len(df_preprocessed))

    # Use pid as index
    #df_preprocessed.set_index("pid", inplace=True)

    ## Save to csv file
    df_preprocessed.to_csv(name + "_preprocessed.csv")


## Read training data, labels, test data
train_features = pd.read_csv('../dataset/train_features.csv', delimiter=',')
test_features = pd.read_csv('../dataset/test_features.csv' , delimiter=',')

#main(train_features, "preprocessed_files/train_features_N_AVG_MIN_MAX_DIFF", is_train=True)
#main(test_features, "preprocessed_files/test_features_N_AVG_MIN_MAX_DIFF", is_train=False)

main(train_features, "preprocessed_files/train_features_N_AVG_MIN_MAX_DIFF_S", is_train=True, use_slopes=True)
main(test_features, "preprocessed_files/test_features_N_AVG_MIN_MAX_DIFF_S", is_train=False, use_slopes=True)
