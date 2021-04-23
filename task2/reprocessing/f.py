import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import shuffle
import datetime
import pickle

## Read training data, labels, test data
train_data = pd.read_csv('../dataset/train_features.csv', delimiter=',')
train_labels = pd.read_csv('../dataset/train_labels.csv', delimiter=',')
test_data = pd.read_csv('../dataset/test_features.csv', delimiter=',')
# sort train data and labels values by pid, so they are the same (because after with the manipulation of data things get nasty)
train_data =train_data.sort_values(['pid','Time'])
train_labels = train_labels.sort_values(['pid'])
test_data =test_data.sort_values(['pid','Time'])
# labels needed for this subtask only up to LABEL_EtCO2
train_labels_1 = train_labels.iloc[:,:11]
# set pid as index for labels (after it will be also for training data because we use the pivot function)
train_labels_1=train_labels_1.set_index('pid')

# First part -> replace nan with the mean of the column
train_data_2_tmp1 = train_data.copy()
test_data_2_tmp1 = test_data.copy()

for col in train_data_2_tmp1.columns:
    #print(col)
    mean = train_data_2_tmp1[col].mean()
    train_data_2_tmp1[col] = train_data_2_tmp1[col].replace(np.nan,train_data_2_tmp1[col].mean())
    #same mean as in the training
    test_data_2_tmp1[col] = test_data_2_tmp1[col].replace(np.nan,test_data_2_tmp1[col].mean())

# normalise the values for each column
for col in train_data_2_tmp1.columns[2:]:
    #print(col)
    minn = train_data_2_tmp1[col].min()
    maxx = train_data_2_tmp1[col].max()
    train_data_2_tmp1[col] = (train_data_2_tmp1[col] - train_data_2_tmp1[col].min()) / (train_data_2_tmp1[col].max() - train_data_2_tmp1[col].min())
    #of the trainind dataset
    test_data_2_tmp1[col] = (test_data_2_tmp1[col] - minn) / (maxx - minn)

# Second part -> add columns with 0 and 1s to indicate if the measurement was taken or not

# change the name of the columns to be able to use the pivot function later
train_data_2_tmp2 = pd.DataFrame()
test_data_2_tmp2 = pd.DataFrame()
for col in train_data.columns[2:]:
    train_data_2_tmp2[col+'_2'] = train_data[col].copy() 
    
for col in test_data.columns[2:]:
    test_data_2_tmp2[col+'_2'] = test_data[col].copy() 

# substitute integers with 1
train_data_2_tmp2=train_data_2_tmp2.mask(train_data_2_tmp2>-9999,1)
test_data_2_tmp2=test_data_2_tmp2.mask(test_data_2_tmp2>-9999,1)
# substitute nan with 0
train_data_2_tmp2=train_data_2_tmp2.replace(np.nan,0)
test_data_2_tmp2=test_data_2_tmp2.replace(np.nan,0)


# concatenate part 1 and part2 for the complete set of training data
train_data_2 = pd.concat([train_data_2_tmp1, train_data_2_tmp2], axis=1)
test_data_2 = pd.concat([test_data_2_tmp1, test_data_2_tmp2], axis=1)# Time Stamp between 1 and 12 (I simply replace it with numbers from 1 to 12) because we need it to use the pivot function after
    
train_data_2['Time']= np.array([[1,2,3,4,5,6,7,8,9,10,11,12] for i in range(int(len(train_data_2['Time'])/12))]).flatten()
test_data_2['Time']= np.array([[1,2,3,4,5,6,7,8,9,10,11,12] for i in range(int(len(test_data_2['Time'])/12))]).flatten()



# now we flatten the dataframe for each patient using the pivot function on the column Time
columns = train_data_2.columns
print(columns[2:])

train_data_2 = train_data_2.pivot(index='pid', columns='Time', values=columns[2:])
test_data_2 = test_data_2.pivot(index='pid', columns='Time', values=columns[2:])

col = 'LABEL_BaseExcess'
train_data_2_0 = train_data_2[train_labels_1[col] == 0.].copy()
train_data_2_1 = train_data_2[train_labels_1[col] == 1.].copy()

min_len = min(len(train_data_2_0),len(train_data_2_1))

train_data_ok = pd.concat([train_data_2_0.sample(min_len),train_data_2_1.sample(min_len)])

train_data_ok = shuffle(train_data_ok)
    
#choose the right labels using the index and .loc function
train_labels_ok = train_labels_1.loc[train_data_ok.index].copy()
print(len(train_labels_ok))
# split train data into train and validation
X_train, X_test, y_train, y_test = train_test_split(train_data_ok,train_labels_ok, train_size=0.8)

