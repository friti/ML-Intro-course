import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn import metrics

##### Loading ######
dfTrain = pd.read_csv('datasets/train.csv', delimiter=',')
X = dfTrain.iloc[:,1:]
Y = dfTrain.iloc[:,0]

## Split in K-folds
## Shuffle
kf = KFold(n_splits=10, shuffle = True)

# Convert to numpy arrays to use indices from KFold later on
X = X.to_numpy()
Y = Y.to_numpy()

# Regularization parameters
reg_params = [0.1, 1, 10, 100, 200]

# Dictionary to store RMSE for each regularization parameter
RMSE = {}

# Loop over all regularisation parameters to test
for reg_param in reg_params:
    
    # Initialize dictionary to store RMSE for each fold as well as avg and std
    RMSE[reg_param] = {"list": [], "avg": 0., "std": 0.}

    # We have to split each time! Cannot loop more than once over kf.split...
    kf_indices = kf.split(X)
    for train_indices, validation_indices in kf_indices:
        # Initialize a Ridge regression model
        ridge_regressor = Ridge(alpha=reg_param)
        # Fit it on the training part of the fold
        ridge_regressor.fit(X[train_indices], Y[train_indices])
        # Evaluate on the rest of the dataset
        y_validation_predictions = ridge_regressor.predict(X[validation_indices])
        # Calculate and store RMSE
        RMSE[reg_param]["list"].append(np.sqrt(metrics.mean_squared_error(Y[validation_indices], y_validation_predictions)))

    # Compute average RMSE and std
    RMSE[reg_param]["avg"] = np.mean(RMSE[reg_param]["list"])
    RMSE[reg_param]["std"] = np.std(RMSE[reg_param]["list"], ddof=1)

file = open("to_submit_shuffle.txt", "w") 

for param in RMSE.keys():
    print("lambda = %.1e" %param)
    print("\tavg = %.3f   std = %.3f" %(RMSE[param]["avg"], RMSE[param]["std"]))
    file.write(str(RMSE[param]["avg"])+"\n")
    
file.close()




