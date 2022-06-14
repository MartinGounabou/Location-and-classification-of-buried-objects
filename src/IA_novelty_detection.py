# %%

from cProfile import label
from code import interact
from statistics import mode
from tkinter.tix import X_REGION
from turtle import color
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psutil import AIX

import scipy as sp
from matplotlib import collections as mc
from scipy import integrate
from sklearn import svm
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, IsolationForest,
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
from sklearn.cluster import KMeans

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

sns.set()

if __name__ == '__main__' :
    
    data = pd.read_csv("DATA\\data{}_dp45_test3.csv".format(0), header=None, index_col=None)
    X1 = np.array(data.iloc[:, 0], dtype=np.float64).reshape(-1,1)
    y1 = np.array( data.iloc[:, 1], dtype=np.float64)
    
    #-------------------------------------------
    
    # X_train, y_train = hlp.standardization(X1[300:]<), hlp.standardization(y1[300:]) 
    # # X_train, y_train = shuffle(X_train, y_train)
    # X_train = np.concatenate((X_train.reshape(-1,1), y_train.reshape(-1,1)), axis = 1)
    
    #-------------------------------------------
    TEST = 1 ;PIPE = 1; Z = 6 ; traj = 6
    
    data = pd.read_csv("DATA\\X_T{}_P{}_Z{}.csv".format(TEST, PIPE, Z), header=None, index_col=None)
    alt = pd.read_csv("DATA\\alt_T{}_P{}_Z{}.csv".format(TEST, PIPE, Z), header=None, index_col=None)
   
    X_train = np.array(data.iloc[:,traj])
    y_train = np.array(alt.iloc[:,traj])
    
    # X_train = np.concatenate(( hlp.standardization(X_train.reshape(-1,1)), hlp.standardization(y_train.reshape(-1,1))), axis = 1)
    X_train =  hlp.standardization(X_train.reshape(-1,1)) - hlp.standardization(y_train.reshape(-1,1))
    # X_train =  hlp.standardization(X_train.reshape(-1,1)) - hlp.standardization(y_train.reshape(-1,1))
    # X_train =  hlp.standardization(X_train.reshape(-1,1)) 
    
# %%

    param_grid = {'kernel' : ['rbf'], 'gamma' : [0.001, 0.01, 0.1, 1], 'nu': [0.001, 0.01, 0.1, 1]}

 
    model = LocalOutlierFactor(novelty=True, n_neighbors=800 )   # 400 for (I-alt) # 1000 for I
    # model =  svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.00005) 
    # model = IsolationForest( )
    # model  = KMeans(n_clusters=2, random_state=0)
 
    # gcv = GridSearchCV(model,param_grid , scoring="accuracy", cv=5,n_jobs=-1, verbose=2)

    # gcv = gcv.fit(X_train, np.array([1]*X_train.shape[0]))

    # model.set_params(**gcv.best_params_)
# %%

    model.fit(X_train)

    y_train = model.predict(X_train)
    
    plt.plot(y_train)
# %%
    TEST = 1
    PIPE = 1
    
    Z = 6
    traj = 5
 
    for traj in range(13) :

        data = pd.read_csv("DATA\\X_T{}_P{}_Z{}.csv".format(TEST, PIPE, Z), header=None, index_col=None)
        alt = pd.read_csv("DATA\\alt_T{}_P{}_Z{}.csv".format(TEST, PIPE, Z), header=None, index_col=None)
        

        # X_test = hlp.lissage(np.array(data.iloc[:,traj]), L=20)
        X_test = np.array(data.iloc[:,traj])
        alt = np.array(alt.iloc[:,traj])
        # offset = [0.137248, 0.13858, 0.138835][int((z_dep-6)/4)]
        f = interpolate.interp1d(y1, X1.ravel(), kind='cubic', fill_value="extrapolate")
        offset = f(Z)
        # ecart = X_test[0] - 0.137248 # ecart pour test z=6
        # ecart = X_test[0] - 0.13858 # ecart pour test z=10
        # ecart = X_test[0] - 0.138835 # ecart pour test z=14
        ecart = X_test[0] - offset
        X_test = X_test - ecart
    
        X_test = np.array(X_test, dtype=np.float64).reshape(-1,1)
    
        # X_test =  hlp.standardization(X_test.reshape(-1,1))  
        X_test =  hlp.standardization(X_test.reshape(-1,1)) - hlp.standardization(alt.reshape(-1,1)) 
        # X_test = np.concatenate(( hlp.standardization(X_test.reshape(-1,1)), hlp.standardization(alt.reshape(-1,1))), axis = 1)
        
        # X_test = shuffle(X_test)

        y_pred = model.predict(X_test)

        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax = ax.flatten()

        # x_val = np.linspace( 40.5, 157, 350)
        x_val = np.linspace( 40, 460, 1271)
        
        ax[0].plot(x_val, X_test, label="X_test = error(current, alt)")
        # ax[0].plot( hlp.standardization(alt.reshape(-1,1)), label="alt")  
        # ax[1].plot(y_pred, label = 'erreur relative')
        ax[1].plot(x_val,  y_pred, label = 'y_pred')
        ax[2].plot(x_val, model.decision_function(X_test), label = 'scores')
        ax[0].legend()
        ax[1].legend( )
        ax[2].legend( )
        
        plt.suptitle("Traj {}, Test {} , Pipe {},  Z0  {}".format(traj+1, TEST, PIPE, Z))
  
        plt.savefig("Traj {}, Test {} , Pipe {},  Z0  {}".format(traj+1, TEST, PIPE, Z))
         
 # %%
