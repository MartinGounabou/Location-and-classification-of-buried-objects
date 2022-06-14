# %%

from code import interact
from turtle import color
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
    
    # data = pd.read_csv("DATA\\data{}_dp45_test3_lissage.csv".format(1), header=None, index_col=None)
    X2 = data.iloc[300:, 0]
    y2 = data.iloc[300:, 1]  

    X2 = np.array(X2, dtype=np.float64).reshape(-1,1)
    y2 = np.array(y2, dtype=np.float64)

    X_train, y_train =  X1[300:],  y1[300:]
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
    sns.set()
    model.fit(X_train , y_train)

    # plt.plot(X2, y2, label = "y_test")
    # plt.plot(X2, y, label = "y_pred")
    # plt.title("Evaluation traj 2, Test 3 ( id 8 ) ")
    # plt.xlabel("I_rms")
    # plt.ylabel("Z ( cm)")
    # hlp.evaluate_model(model, X_train, y_train, X_test, y_test, verbose=True)
    # plt.legend()
    

# %%
    TEST = 1
    PIPE = 1
    Z = 6
    traj = 5
    for Z in [6, 10, 14]: 
 
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

            y_pred = model.predict(X_test)
            w = 1271
            y_pred = y_pred[:w]
            alt = alt[:w]
            

            # plt.figure()
            # plt.plot(X_test, label="".format(traj))
            
            fig, ax = plt.subplots(1,2, figsize=(12,4))
            ax = ax.flatten()

            # x_val = np.linspace( 40.5, 157, 350)
            x_val = np.linspace( 40, 460, 1271)
            
            # y_pred =  hlp.standardization(y_pred)
            # alt =   hlp.standardization(alt)

            
            ax[0].plot(x_val,  y_pred, label="y_pred")
            ax[0].plot(x_val,  alt, label="alt")  
            
            ax[0].set_xlabel(' X (cm) ')
            ax[0].set_ylabel(' Z (cm) ')
            ax[0].legend(loc='right')

            ax[1].plot(x_val , (y_pred-alt)/alt, label = 'erreur')
            # ax[1].plot(x_val , (y_pred-alt)/alt, label = 'erreur relative')
            # ax[1].plot(x_val , X_test, label = 'erreur relative')
            


            ax[1].set_xlabel(' X (cm) ')
            ax[1].set_ylabel(' Z (cm) ')

            plt.suptitle("Traj {}, Test {} , Pipe {},  Z0  {}".format(traj+1, TEST, PIPE, Z))
            plt.legend()
            
            
            plt.savefig("Traj {}, Test {} , Pipe {},  Z0  {}".format(traj+1, TEST, PIPE, Z))
         
        
# %%   
    """
   

    # plt.figure()
    # plt.scatter(X_test, alt)
    # plt.title("Evaluation de l'altitude en fonction du courant ")
    # plt.xlabel("I_rms")
    # plt.ylabel("Z (cm)")
    # # hlp.evaluate_model(model, X_train, y_train, X_test, y_test, verbose=True)
    # plt.legend()
    # plt.savefig("Traj {}, Test {} , Pipe {},  Z0  {}".format(traj+1, TEST, PIPE, Z))
    """
    
# from scipy import interpolate
# f = interpolate.interp1d(X_train, y_train, fill_value="extrapolate")
# y = f(X_test.ravel())
 
    
# %%

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
    
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax = ax.flatten()

    # ax[0].plot(np.linspace( 40.5, 157, 350), y_pred, label="y_pred")
    # ax[0].plot(np.linspace( 40.5, 157, 350), alt , label="alt")
    
    
    alt = np.interp( np.linspace( 40, 370, 1271), np.linspace( 40, 460, 1271), alt )

    b = 0
    a = 200
    
    
    c = np.linspace( 40, 460, 1271)[b:a].reshape(-1,1)
    d1 = hlp.standardization(y_pred[b:a]).reshape(-1,1)
    d2 = hlp.standardization(alt[b:a]).reshape(-1,1)
    
    
    b = 200
    a = 1271
    
    c = np.concatenate((c, np.linspace( 40, 460, 1271)[b:a].reshape(-1,1)), axis = 0)
    d1 = np.concatenate((d1, hlp.standardization(y_pred[b:a]).reshape(-1,1)), axis = 0)
    d2 = np.concatenate((d2, hlp.standardization(alt[b:a]).reshape(-1,1)), axis = 0)
    
    # ax[0].plot(np.linspace( 40, 460, 1271)[b:a], hlp.standardization(y_pred[b:a]), label="y_pred")
    # ax[0].plot(np.linspace( 40, 460, 1271)[b:a],  hlp.standardization(alt[b:a]), label="alt")  
        
    ax[0].plot(c, d1, label="y_pred")
    ax[0].plot(c, d2, label="alt")   
        
    
    ax[0].set_xlabel(' X (cm) ')
    ax[0].set_ylabel(' Z (cm) ')
    ax[0].legend(loc='right')

    # ax[1].plot(np.linspace( 40.5, 157, 350) , (y_pred-alt)/alt, label = 'erreur relative')
    # ax[1].plot(np.linspace( 40, 370, 1271) , (y_pred-alt)/alt, label = 'erreur relative')
    ax[1].plot(c , (d1-d2)/d2, label = 'erreur relative')
    
    ax[1].set_xlabel(' X (cm) ')
    ax[1].set_ylabel(' Z (cm) ')

    plt.suptitle("Traj {}, Test {} , Pipe {},  Z0  {}".format(traj+1, TEST, PIPE, Z))
    plt.legend()