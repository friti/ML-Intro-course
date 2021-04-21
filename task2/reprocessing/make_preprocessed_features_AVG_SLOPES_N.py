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
    feature_names = [ x for x in df.columns  if x not in ("pid", "Time") ]
    print("Features:")
    print(feature_names)


    ## Replace the series of 12 measurements by their average and slope
    print(datetime.datetime.now())

        
    counts = df[["pid"]+feature_names].groupby(["pid"]).count().add_suffix("_n")
    avgs = df[["pid"]+feature_names].groupby(["pid"]).mean().add_suffix("_avg")
    df_preprocessed = pd.concat([counts, avgs], axis=1)

    if use_slopes:
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
            avgs = df_preprocessed[feature_name + "_avg"].replace(np.nan, 0)
            avg_avg = np.average(avgs, weights=df_preprocessed[feature_name + "_n"])
            print(feature_name + "_avg")
            print("Avg avg over not NaN  : %.3f" %(avg_avg))
            imputation[feature_name + "_avg"] = avg_avg
            df_preprocessed[feature_name + "_avg"].replace(np.nan, avg_avg, inplace=True)

            if use_slopes:
                slopes = df_preprocessed[feature_name + "_slope"].replace(np.nan, 0)
                slope_avg = np.average(slopes, weights=df_preprocessed[feature_name + "_n"])
                print(feature_name + "_slope")
                print("Avg slope over not NaN: %.3f" %(slope_avg))
                imputation[feature_name + "_slope"] = slope_avg
                df_preprocessed[feature_name + "_slope"].replace(np.nan, slope_avg, inplace=True)

        with open('imputation.json', 'w') as f:
           json.dump(imputation, f)

    else:
        with open('imputation.json') as f:
            imputation = json.load(f)

        for feature_name in feature_names:
            avg_avg = imputation[feature_name + "_avg"]
            df_preprocessed[feature_name + "_avg"].replace(np.nan, avg_avg, inplace=True)

            if use_slopes:
                slope_avg = imputation[feature_name + "_slope"]
                df_preprocessed[feature_name + "_slope"].replace(np.nan, slope_avg, inplace=True)


    ## Normalize features
    if is_train:
        normalisation = {}
        for feature_name in feature_names:
            # Std scaling
            df_preprocessed[feature_name + "_avg"], avg1, std1 = std_scaler(df_preprocessed[feature_name + "_avg"])
            print(feature_name + "_avg")
            print("mean: %.3f   std: %.3f" %(avg1, std1))
            normalisation[feature_name + "_avg"] = {"mean": avg1, "std": std1}

            if use_slopes:
                df_preprocessed[feature_name + "_slope"], avg2, std2 = std_scaler(df_preprocessed[feature_name + "_slope"])
                print(feature_name + "_slope")
                print("mean: %.3f   std: %.3f" %(avg2, std2))
                normalisation[feature_name + "_slope"] = {"mean": avg2, "std": std2}

        with open('normalisation.json', 'w') as f:
           json.dump(normalisation, f)

    else:
        with open('normalisation.json') as f:
            normalisation = json.load(f)
        for feature_name in feature_names:
            var = feature_name + "_avg"
            df_preprocessed[var] = scale(df_preprocessed[var], normalisation[var]["mean"], normalisation[var]["std"])
            if use_slopes:
                var = feature_name + "_slope"
                df_preprocessed[var] = scale(df_preprocessed[var], normalisation[var]["mean"], normalisation[var]["std"])
            

    print(df_preprocessed.head())
    print(len(df_preprocessed))

    # Use pid as index
    #df_preprocessed.set_index("pid", inplace=True)

    ## Save to csv file
    df_preprocessed.to_csv(name + "_preprocessed.csv")


## Read training data, labels, test data
train_features = pd.read_csv('dataset/train_features.csv', delimiter=',')
test_features = pd.read_csv('dataset/test_features.csv' , delimiter=',')


main(train_features, "train_features", is_train=True)
main(test_features, "test_features", is_train=False)
