import pandas as pd
import numpy as np
import datetime
import pickle
import json
import sys


def make_linear_fit(x, y):
    x = x.where(y.notna()).dropna()
    y = y.dropna()
    if len(x) < 2:
        return 0.
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

    print(df)


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
    feature_names = [ "RRate", "ABPm", "SpO2", "Heartrate" ]
    print("Features:")
    print(feature_names)

    df_preprocessed = df.copy()

    ## Normalize features (Age will be done later)
    if is_train:
        normalisation = {}
        for feature_name in feature_names:
            # Std scaling
            df_preprocessed[feature_name], avg, std = std_scaler(df_preprocessed[feature_name])
            print("%s mean: %.3f   std: %.3f" %(feature_name, avg, std))
            normalisation[feature_name] = {"mean": avg, "std": std}
            # normalisation json data saved when normalizing age
    else:
        with open(name.replace("test", "train") + '_normalisation.json') as f:
            normalisation = json.load(f)
        for feature_name in feature_names:
            df_preprocessed[feature_name] = scale(df_preprocessed[feature_name], normalisation[feature_name]["mean"], normalisation[feature_name]["std"])

    print(df_preprocessed)

    ## Replace NaNs by 0
    for feature_name in feature_names:
        df_preprocessed[feature_name].replace(np.nan, 0, inplace=True)


    ## Enforce Time to go from 1 to 12
    df_preprocessed['Time'] = np.array([ list(range(1,13)) for i in range(len(pids))]).flatten()


    # Pivot dataframe
    df_preprocessed = df_preprocessed.pivot(index='pid', columns='Time', values=feature_names)
    # Convert to dataframe
    df_preprocessed = pd.DataFrame(df_preprocessed.to_records())
    # Repair the column names
    df_preprocessed.columns = [ col[2:-1].replace("', ", "_") if col != "pid" else col for col in df_preprocessed.columns]
    # Use pid as index
    df_preprocessed.set_index("pid", inplace=True)
    print(df_preprocessed)


    ## Add Age and counts
    ages = df[["pid", "Age"]].groupby(["pid"]).mean()
    counts = df[["pid"]+feature_names].groupby(["pid"]).count().add_suffix("_n")
    df_preprocessed = pd.concat([ages, df_preprocessed, counts], axis=1)

    ## Normalization for Age
    if is_train:
        feature_name_full = "Age"
        df_preprocessed[feature_name_full], avg, std = std_scaler(df_preprocessed[feature_name_full])
        print("%s mean: %.3f   std: %.3f" %(feature_name_full, avg, std))
        normalisation[feature_name_full] = {"mean": avg, "std": std}

        with open(name + '_normalisation.json', 'w') as f:
           json.dump(normalisation, f)
    else:
        # Age
        feature_name_full = "Age"
        df_preprocessed[feature_name_full] = scale(df_preprocessed[feature_name_full], normalisation[feature_name_full]["mean"], normalisation[feature_name_full]["std"])


    ## Add slopes
    if use_slopes:
        print("Slope started at %s" %(datetime.datetime.now()))
        df_preprocessed["time_list"] = df.groupby(["pid"]).Time.apply(list)
        for feature_name in feature_names:
            df_preprocessed[feature_name + "_list"] = df.groupby(["pid"])[feature_name].apply(list)
            df_preprocessed[feature_name + "_slope"] = df_preprocessed.apply(lambda row: make_linear_fit(pd.Series(row["time_list"]), pd.Series(row[feature_name + "_list"])), axis=1)
            # Delete the _list columns
            df_preprocessed.drop(feature_name + "_list", axis=1, inplace=True)

            print("%s finished at %s" %(feature_name, datetime.datetime.now()))
            
        # Delete the time_list columns
        df_preprocessed.drop("time_list", axis=1, inplace=True)
        print("Slope finished at %s" %(datetime.datetime.now()))
 

    print("=======================")
    print("=== FINAL DATAFRAME ===")
    print("=======================")
    print("")
    print(df_preprocessed)

    print("")
    print("")

    ## Save to csv file
    df_preprocessed.to_csv(name + "_preprocessed.csv")


## Read training data, labels, test data
train_features = pd.read_csv('../dataset/train_features.csv', delimiter=',')
test_features = pd.read_csv('../dataset/test_features.csv' , delimiter=',')

#main(train_features, "preprocessed_files/train_features_SERIES_IMPUTE0", is_train=True)
#main(test_features, "preprocessed_files/test_features_SERIES_IMPUTE0", is_train=False)

main(train_features, "preprocessed_files/train_features_SERIES_IMPUTE0_N_S", is_train=True, use_slopes=True)
main(test_features, "preprocessed_files/test_features_SERIES_IMPUTE0_N_S", is_train=False, use_slopes=True)
