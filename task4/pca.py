import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn import preprocessing
from sklearn import decomposition


features_directory = "./preprocessed_features/"
figures_directory = "./pca/"

def study(model_name):

    ## Load data
    print("Loading and normalizing data...")
    X = pd.read_csv(features_directory + model_name + "_features.csv", sep=",")

    ids = X["id"]
    X = X.drop(["id"], axis=1)                                      # remove id column
    X = preprocessing.StandardScaler().fit_transform(X)             # scale features so that they have an
                                                                    # average 0 and std dev 1: f <- (f-f*)/s(f) 
    n_features = X.shape[1]


    ## Principal Component analysis
    print("\nPrincipal Component Analysis...")
    pca = decomposition.PCA(n_components=n_features)
    print("Fit...")
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    print("Principal component\t% of variance explained\tCumulative sum")
    explained_variance_ratio_lite = explained_variance_ratio[:10]
    for i,(r,rc) in enumerate(zip(explained_variance_ratio_lite, explained_variance_ratio_lite.cumsum())):
        print("%d\t\t\t%.3f\t\t\t%.3f" %(i+1,r,rc))


    ## Features in Principal Component basis
    pc_featBasis = pca.components_              # Principal axes in feature space
    feat_pcBasis = np.transpose(pc_featBasis)   # Features in principal axes space

    # Only look at the most explanatory components
    pc_featBasis = pc_featBasis[:, :20]

    plt.figure()

    plt.title("Principal Components in feature basis")
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(pc_featBasis, annot=False, cbar=True)

    plt.savefig(figures_directory + model_name + "_principle_components_in_feature_basis.pdf")
    plt.close()


    ## Explained variance as a function as the number of PC
    plt.figure()

    plt.title("Cumulative explained variance as a function\nof the number of Principal Components")
    plt.plot( [i for i in range(0,n_features+1)], [0]+list(explained_variance_ratio.cumsum()),
              color="steelblue",
              linestyle="-" )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative explained variance")
    plt.xlim([0, n_features])
    plt.ylim([0, 1])

    plt.savefig(figures_directory + model_name + "_explained_variance_cumulative.pdf")
    plt.close()


    ## Explained variance for each PC
    plt.figure()

    plt.title("Explained variance for each Principal Component")
    plt.plot( [i for i in range(1,n_features+1)], list(explained_variance_ratio),
              color="steelblue",
              linestyle="-" )
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained variance")
    plt.xlim([0, n_features])
    plt.ylim([0, 0.35])

    plt.savefig(figures_directory + model_name + "_explained_variance_perComponent.pdf")
    plt.close()


    ## Compute the new features from PCA
    principle_component_features = pca.transform(X)

    return ids, principle_component_features


def make_principle_component_features_file(model_name, ids, principle_component_features):

    # Initialize output CSV file
    n_features = principle_component_features.shape[1]
    filename = features_directory + model_name + "_PCfeatures.csv"
    csvfile = open(filename, 'w')
    csvwriter = csv.writer(csvfile) 

    featureNames = [ "id" ] + [ "f"+str(x+1) for x in range(n_features) ]
    csvwriter.writerow(featureNames) 

    # Make n dim array with id and pc features
    ids = np.array([[x] for x in ids])    # necessary manipulation for concatenation
    data = np.concatenate((ids, principle_component_features), axis=1)

    # Write features to csv file
    csvwriter.writerows(data)

    # Close the csv file
    csvfile.close()

    return


if __name__ == "__main__":

    model_name = "ResNet50"
    ids, principle_component_features = study(model_name)
    make_principle_component_features_file(model_name, ids, principle_component_features)
