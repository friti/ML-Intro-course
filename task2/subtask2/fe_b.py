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

sel1 = False
#use only the specific column to train
#onecol = True
#age = True
#use the avg and slope for the training
age = False
age_n = False
age_n_avg = True
age_n_avg_diff = False
age_n_slope = False
onlyn = False
n_avg = False
age_n_avg_slope = False
#n_avg_diff = False
#n_avg_diff_slope = True

folder = 'modelB_balanced'+str(balanced)+'_allcol_agenavgdiff/'
plt.ioff()

os.system('mkdir -p %s' %folder)
os.system('mkdir -p %s/conf_mat' %folder)

print("Balanced = ", balanced,"; Lite = no")
train_features_original = pd.read_csv('../reprocessing/preprocessed_files/train_features_N_AVG_MIN_MAX_DIFF_S_preprocessed.csv')
#train_features_original = pd.read_csv('../reprocessing/preprocessed_files/train_features_N_AVG_MIN_MAX_DIFF_S_norm01_preprocessed.csv')
train_labels_original = pd.read_csv('../dataset/train_labels.csv')

#sort the train_labels by pid
train_labels_original = train_labels_original.sort_values(['pid'])

train_features_original.index = train_features_original["pid"]
train_labels_original.index = train_labels_original["pid"]
print(train_features_original)
print(train_labels_original)

'''no_feat = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
ok_feat= []
for col in train_features_original.columns:
    flag = 0
    for f in no_feat:
        if f.split("LABEL_")[1] in col:
            flag = 1
    if flag == 1:
        continue
    elif col == 'pid':
        continue
    else:
        ok_feat.append(col)
#ok_feat = [ col for col in train_features_original.columns if no_feat.split("LABEL_")[0] not in col]
'''
label_features = ['LABEL_Sepsis']

models = []
mean = 0.
for lab in label_features:
    print("Feature:",lab)
    lab_ok = lab.split("LABEL_")[1]
    train_features = train_features_original.iloc[:,1:].copy()
    train_labels = train_labels_original[lab].copy()

    '''if sel1:
        train_features =train_features[ok_feat].copy()
        print(train_features.columns)
    '''
    if age:
        list_feat = [col for col in train_features.columns if ("Age" in col)]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if age_n:
        list_feat = [col for col in train_features.columns if ("_n" in col or "Age" in col)]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if onlyn:
        list_feat = [col for col in train_features.columns if ("_n" in col)]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if age_n_avg:
        list_feat = [col for col in train_features.columns if ("_n" in col or "Age" in col or "_avg" in col)]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if n_avg:
        list_feat = [col for col in train_features.columns if ("_n" in col  or "_avg" in col)]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if age_n_avg_diff:
        list_feat = [col for col in train_features.columns if ("_n" in col or "Age" in col or "_avg" in col or "_diff" in col)]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if age_n_slope:
        list_feat = [col for col in train_features.columns if ("_n" in col or "Age" in col or "_slope" in col )]
        train_features = train_features[list_feat]
        print("Chosen columns for the training:",train_features.columns)

    if age_n_avg_slope:
        list_feat = [col for col in train_features.columns if ("_n" in col or "Age" in col or "_slope" in col or "_avg" in col )]
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
    filename = folder + 'model_balanced'+str(balanced)+'_22Apr_'+lab_ok +'.sav'
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
