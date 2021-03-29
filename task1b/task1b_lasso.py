#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics


## Read training data
train_data = pd.read_csv('datasets/train.csv', delimiter=',')
X = train_data.iloc[:,2:]
y = train_data.iloc[:,1]

## Check data
print("\n===== CHECK DATA =====")
print("TRAIN DATA")
print(train_data.head())
print("\n")
print("ORIGINAL FEATURES")
print(X.head())
print("\n")
print("TARGET")
print(y.head())

# Linear features are already the x1, ..., x5

# Quadratic features
X["x6"] = X["x1"]**2
X["x7"] = X["x2"]**2
X["x8"] = X["x3"]**2
X["x9"] = X["x4"]**2
X["x10"] = X["x5"]**2

# Exponential features
X["x11"] = np.exp(X["x1"])
X["x12"] = np.exp(X["x2"])
X["x13"] = np.exp(X["x3"])
X["x14"] = np.exp(X["x4"])
X["x15"] = np.exp(X["x5"])

# Cosine features
X["x16"] = np.cos(X["x1"])
X["x17"] = np.cos(X["x2"])
X["x18"] = np.cos(X["x3"])
X["x19"] = np.cos(X["x4"])
X["x20"] = np.cos(X["x5"])

# Constant
X["x21"] = 1

# Check new features
print("NEW FEATURES")
print(X.head())


## Optimize Lasso regression parameter with a Grid Search
#  Re-used sample code at https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html
lasso = Lasso(max_iter=100000, fit_intercept=False)
alphas = np.logspace(-4, -0.5, 30) #study alphas in this range

tuned_parameters = [{'alpha': alphas}]
n_folds = 7   # 100 events per fold

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=True)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

# Make figure
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
plt.savefig("alpha_grid_search.pdf")


## Best regression parameter alpha
alpha_best = alphas[np.argmax(scores)]
print("Best regression parameter: %.2e" %alpha_best)


## Train Lasso model for the best regression parameter on the whole dataset
model = Lasso(alpha=alpha_best, fit_intercept=False)
model.fit(X, y)
# Print coefficients
print(model.coef_)


## Make file to submit
file = open("to_submit.txt", "w") 
for coef in model.coef_:
    file.write(str(coef)+"\n")
file.close()
