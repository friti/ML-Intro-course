# Linear Regression
# First compute the required features and after to a linear regression of them
import pandas as pd
import numpy as np
import csv

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dfTrain = pd.read_csv('datasets/train.csv', delimiter=',')
X_raw = dfTrain.iloc[:,2:]
y = dfTrain.iloc[:,1]

#Compute the features for each entry
#linear feature
X_lin = X_raw 
#quadratic feature
X_sq = np.square(X_raw)
#exponential feature
X_exp = np.exp(X_raw)
#cosine feature
X_cos = np.cos(X_raw)

# concatenate
X = pd.concat([X_lin, X_sq, X_exp, X_cos], axis=1)

# add the constant column
X['phi21'] = np.ones(len(X_raw['x1']))

#Now that we have all the right features, we split into train and test data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.8)

# fit the model = linear regression
model = LinearRegression()
model.fit(XTrain, yTrain)

#prediction and RMSE
yTestPred = model.predict(XTest)
print("RMSE: %.2e" %(np.sqrt(metrics.mean_squared_error(yTest, yTestPred))))

#coefficients from the linear regression
coefficients = model.coef_
#print(coefficients)
#print("the lenght is ",len(coefficients))

# in the csv submit file we need to put the coefficients of the linear regression
df = pd.DataFrame(data={"col1": coefficients})
df.to_csv("submit_fe.csv", sep=',',index=False, header=None)




