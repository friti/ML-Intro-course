import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import shuffle
import pickle
import os

balanced = False

#use lite sample
lite = False

#use only the specific column to train
onecol = True
age = True
#use the avg and slope for the training
n_avg = False
n_avg_diff = False
n_avg_diff_slope = True

folder = 'modelB_balanced'+str(balanced)+'_onecol_agenavgdiffslope/'
plt.ioff()

os.system('mkdir -p %s' %folder)
os.system('mkdir -p %s/conf_mat' %folder)

print("Balanced = ", balanced,"; Lite = no")
train_features_original = pd.read_csv('../reprocessing/preprocessed_files/train_features_N_AVG_MIN_MAX_DIFF_S_preprocessed.csv')
train_labels_original = pd.read_csv('../dataset/train_labels.csv')

#sort the train_labels by pid
train_labels_original = train_labels_original.sort_values(['pid'])

train_features_original.index = train_features_original["pid"]
train_labels_original.index = train_labels_original["pid"]
print(train_features_original)
print(train_labels_original)

label_features = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
#label_features = ['LABEL_BaseExcess']

models = []
mean = 0.
for lab in label_features:
    print("Feature:",lab)
    lab_ok = lab.split("LABEL_")[1]

    #choose only the respective column for training
    if onecol:
        if age:
            train_features = pd.concat([train_features_original["Age"],train_features_original.filter(like=lab_ok).copy()],axis = 1)
        else:
            train_features =train_features_original.filter(like=lab_ok).copy()
        train_labels = train_labels_original[lab].copy()
    else:
        train_features = train_features_original.copy()
        train_labels = train_labels_original[lab].copy()
        
    if lite :
        #lite vestion to speed up
        pids = train_features_original["pid"].drop_duplicates().to_list()
        Npatients = len(pids)//10
        Npatients = Npatients
        pids = pids[:Npatients]
        train_features = train_features[train_features.index.isin(pids)]
        train_labels = train_labels[train_labels.index.isin(pids)]
        
        print("Lite number of patients: %d" %Npatients)                                

        print("lenghts of training set original and lite:",len(train_features_original),len(train_features))
    
    if n_avg :
        list_feat = [lab_ok+'_n',lab_ok+'_avg']
        if age:
            list_feat = ["Age"] + list_feat
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    elif n_avg_diff:
        list_feat = [lab_ok+'_n',lab_ok+'_avg',lab_ok+'_diff']
        if age:
            list_feat = ["Age"] + list_feat
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    elif n_avg_diff:
        list_feat = [lab_ok+'_n',lab_ok+'_avg',lab_ok+'_diff',lab_ok+'_slope']
        if age:
            list_feat = ["Age"] + list_feat
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)


    if balanced:
        #BALANCED
        train_features_0 = train_features[train_labels == 0.].copy()
        train_features_1 = train_features[train_labels == 1.].copy()
        
        #find the min lenght
        min_len = min(len(train_features_0),len(train_features_1))
        print("min len",min_len)
    
        # balanced training data sample
        train_features = pd.concat([train_features_0.sample(min_len),train_features_1.sample(min_len)])
        print("Lenght of the balanced training set",train_features.shape[0], " ; Initial lenght ",train_features_original.shape[0])
        #shuffle the sample
        train_features = shuffle(train_features)
        print("In the balanced case:")
        #print(train_features)
        #print(train_labels)
        #choose the right labels using the index and .loc function
        train_labels = train_labels.loc[train_features.index].copy()
        #print(train_labels)
    

    X_train, X_test, y_train, y_test = train_test_split(train_features,train_labels, train_size=0.8)
    print("Fit Start time")
    print(datetime.datetime.now())

    classifier = svm.SVC(probability = True)
    classifier.fit(X_train,y_train)
    print("Fit End  time")
    print(datetime.datetime.now())

    #save model into file
    filename = folder + 'model_balanced'+str(balanced)+'_12columns_22Apr_'+lab_ok +'.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    models.append(classifier)
    
    y_pred = classifier.predict_proba(X_test)
    y_pred = pd.DataFrame(data=y_pred, columns=['prob_0','prob_1'])
    print("Column: "+lab+" ROC AUC: %.2f" %(metrics.roc_auc_score(y_test, y_pred['prob_1'])))
    mean += metrics.roc_auc_score(y_test, y_pred['prob_1'])
    matrix = plot_confusion_matrix(classifier, X_test, y_test)
    plt.title('Confusion matrix for classifier')
    plt.savefig(folder+'conf_mat/'+lab_ok+'.png')

mean /= len(label_features)
print("ROC AUC mean:",mean)
