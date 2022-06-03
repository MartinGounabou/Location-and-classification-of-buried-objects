# %%

from code import interact
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy as sp
from matplotlib import collections as mc
from scipy import integrate
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, make_scorer,
                             mean_absolute_error, mean_squared_error, median_absolute_error,
                             plot_confusion_matrix, precision_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from scipy import interpolate

import utils.augmentation as aug
import utils.helper as hlp
from altitude_manage import Data_extraction
import scipy.ndimage.filters as filters
import seaborn as sns


if __name__ == '__main__' :
    
    data = pd.read_csv("DATA\\data{}_dp45_test3.csv".format(0), header=None, index_col=None)
    

    X1 = data.iloc[:, 0]
    y1 = data.iloc[:, 1]

    X1 = np.array(X1, dtype=np.float64).reshape(-1,1)
    y1 = np.array(y1, dtype=np.float64)
    
    data = pd.read_csv("DATA\\data{}_dp45_test3_lissage.csv".format(1), header=None, index_col=None)
    X2 = data.iloc[300:, 0]
    y2 = data.iloc[300:, 1]  

    X2 = np.array(X2, dtype=np.float64).reshape(-1,1)
    y2 = np.array(y2, dtype=np.float64)

    X_train, y_train = X1[300:], y1[300:] 
    # X_train, y_train = shuffle(X_train, y_train)
    
    
    param_grid = {
    'n_estimators': [10,50,100],
    'criterion': ['mse', 'mae'],
    'max_depth': [2,8,16,32,50],
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2],
    #'oob_score': [True, False],
    'max_features': ['auto','sqrt','log2'],    
    'bootstrap': [True, False],
    'warm_start': [True, False],
}
    
    model = ExtraTreesRegressor()

    # gcv = GridSearchCV(model,param_grid,cv=5,n_jobs=-1, verbose=2).fit(X_train.reshape(-1,1), y_train)

    # model.set_params(**gcv.best_params_)
#%%
    model.fit(X_train , y_train)
    y = model.predict(X2)
    plt.plot(X2, y2)
    plt.plot(X2, y)


    

# %%
    TEST = 2
    PIPE = 3
    
    Z = 12
    traj =12
    
    data = pd.read_csv("DATA\\X_T{}_P{}_Z{}.csv".format(TEST, PIPE, Z), header=None, index_col=None)
    alt = pd.read_csv("DATA\\alt_T{}_P{}_Z{}.csv".format(TEST, PIPE, Z), header=None, index_col=None)
    
    X_test = np.array(data.iloc[:,traj])
    alt = np.array(alt.iloc[:,traj])


    # offset = [0.137248, 0.13858, 0.138835][int((z_dep-6)/4)]
    
    f = interpolate.interp1d(y1, X1.ravel(), kind='linear', fill_value="extrapolate")
    offset = f(Z)
    
    # ecart = X_test[0] - 0.137248 # ecart pour test z=6
    # ecart = X_test[0] - 0.13858 # ecart pour test z=10
    # ecart = X_test[0] - 0.138835 # ecart pour test z=14
    
    ecart = X_test[0] - offset
    X_test = X_test - ecart
    

    X_test = np.array(X_test, dtype=np.float64).reshape(-1,1)

    y_pred = model.predict(X_test.reshape(-1,1))
    
    w = 1271
    y_pred = y_pred[:w]
    alt = alt[:w]
    
    # plt.figure()
    # plt.plot(X_test, label="".format(traj))
    plt.figure()
    plt.plot(  y_pred, label="y_pred")
    # plt.figure()
    plt.plot(alt , label="alt")
    plt.figure()
    plt.plot((y_pred-alt))
    plt.legend()

    plt.figure()
    plt.plot(X_test, alt)
# from scipy import interpolate
# f = interpolate.interp1d(X_train, y_train, fill_value="extrapolate")
# y = f(X_test.ravel())
# %%

    X_test = X_test[:200]
    alt = alt[:200]
    id = alt.argsort()
    alt = alt[id]
    X_test = X_test[id]
    plt.scatter(alt, X_test)
    
# %%