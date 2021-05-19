## OS
import os


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
from sklearn.metrics import f1_score


## Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation


def main():
    ## Load training dataset
    train_df = pd.read_csv('dataset/train.csv', delimiter=',')

    print("\n=== Initial DataFrame ===")
    print(train_df.head())


    ## Encoder for the dataframe
    amino_acids = ["R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "U", "G", "P", "A", "I", "L", "M", "F", "W", "Y", "V"]

    def one_hot_encoding(df, amino_acids):    
        
        ## Columns to be used for training
        training_columns = []
        
        ## Make a column for each mutation
        for site_number in range(4):
            site_column = "site" + str(site_number+1)
            df[site_column] = df["Sequence"].apply(lambda x: x[site_number])
            
        ## One hot encode all columns
        for site_number in range(4):
            column = "site" + str(site_number+1)
            for amino_acid in amino_acids:
                new_column = column + "_" + amino_acid
                df[new_column] = (df[column] == amino_acid).astype(int)
                training_columns.append(new_column)
        
        return training_columns


    train_df_encoded = train_df.copy()
    training_columns_encoded = one_hot_encoding(train_df_encoded, amino_acids)

    print("\n=== Encoded DataFrame ===")
    print(train_df_encoded.head())


    ## Make features and predictions datasets
    X = train_df_encoded[training_columns_encoded]
    y = train_df_encoded["Active"]
    y_2D = pd.DataFrame({"p0": y, "p1":y.apply(lambda x: not x).astype(int)})

    print("\n=== Features and labels ===")
    print("Features:")
    print(X.head())

    print("\nLabels:")
    print(y.head())

    print("\nLabels 2D:")
    print(y_2D.head())


    ## Evaluating class_imbalance
    print("\n=== Class imbalance ===")
    w0 = 1.
    w1 = sum(y==0)/sum(y==1)
    print("w0 = %.3f" %w0)
    print("w1 = %.3f" %w1)
    class_weights = {0: w0, 1: w1}


    print("\n=== Training model ===")

    ## Train test split
    print("\nSplitting into training dataset (80%), validation dataset (10%) and test dataset (10%)...")
    X_train, X_vt, y_2D_train, y_2D_vt = train_test_split(X, y_2D, train_size=0.80)
    X_validation, X_test, y_2D_validation, y_2D_test = train_test_split(X_vt, y_2D_vt, train_size=0.5)
    y_test = y_2D_test.p0


    ## NN model
    print("\nDefining model...")
    activation = 'relu'

    model = Sequential()

    model.add(Dense(300, input_dim=X_train.shape[1]))
    model.add(Activation(activation))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200))
    model.add(Activation(activation))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense( 2, activation='softmax', name='output'))


    #model.add(Dense(400, input_dim=X_train.shape[1]))
    #model.add(Activation(activation))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    #model.add(Dense(100))
    #model.add(Activation(activation))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    #model.add(Dense( 2, activation='softmax', name='output'))

    print("Model summary:")
    print(model.summary())


    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['acc', 'AUC'],
    )

    print("\nFitting...")
    history = model.fit(X_train, y_2D_train,
                        validation_data=(X_validation, y_2D_validation),
                        class_weight=class_weights,
                        epochs=42,
                        batch_size=1000)


    ## Prediction on the test dataset
    print("\nPredictions for the test sample...")
    y_pred_proba = model.predict(X_test)
    cuts = np.logspace(-8, 1, 100)
    y_preds = [ y_pred_proba[:,0]>=cut for cut in cuts ]
    f1_scores = [ f1_score(y_test, y_preds[icut]) for icut in range(len(cuts)) ]


    ## F1 score as a fct of cut
    plt.figure()
    plt.plot(cuts, f1_scores)
    plt.xlabel("Cut value")
    plt.ylabel("F1 score")
    plt.xscale("log")
    plt.savefig("f1_score.pdf")
    plt.close()


    ## Best F1 score
    best_cut_idx = np.argmax(f1_scores)
    best_cut = cuts[best_cut_idx]
    y_pred = y_preds[best_cut_idx]

    print(y_pred)
    print("F1 score: %.3f" %(f1_score(y_test, y_pred)))


    def plot_var(variable, history):
        plt.title(variable)
        plt.plot(history.history[variable][2:], label='train')
        plt.plot(history.history['val_'+variable][2:], label='validation')
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.ylabel(variable)
        plt.grid(True)



    ## Loss
    plt.figure()
    plot_var("loss", history)
    plt.savefig("loss.pdf")
    plt.close()


    ## AUC
    plt.figure()
    plot_var("auc", history)
    plt.savefig("auc.pdf")
    plt.close()


    ## Load test dataset
    test_df = pd.read_csv('dataset/test.csv', delimiter=',')
    print("\nPredictions for the test dataset...")

    test_df_encoded = test_df.copy()
    columns_encoded = one_hot_encoding(test_df_encoded, amino_acids)
    X_test = test_df_encoded[columns_encoded]

    y_pred_test_proba = model.predict(X_test)
    y_pred_test = y_pred_test_proba[:,0]>=best_cut

    np.savetxt("submit.txt", y_pred_test, fmt="%d")


    return


if __name__ == "__main__":
    main()
