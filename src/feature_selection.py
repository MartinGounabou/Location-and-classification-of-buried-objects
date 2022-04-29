# %%



from doctest import FAIL_FAST
from tkinter.tix import Tree
from sklearn import tree
from sympy import trunc
from data_manipulation_burried_object_localisation import Data_extraction
import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
import  seaborn as sns
import pylab as pl
from matplotlib import collections as mc
from collections import OrderedDict

# ______________sckitlearn

from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.inspection import permutation_importance
# --------------- utils

import utils.augmentation as aug
import utils.helper as hlp


class Artificial_intelligence(Data_extraction):

    def __init__(self) -> None:
        super().__init__(ESSAI=2, TEST=2)
        self.path_to_data = os.path.join(
            self.path_to_data_dir, 'data_all_z_dp13.csv')
        self.path_to_features = os.path.join(
            self.path_to_data_dir, "features.csv")
        self.path_to_labels = os.path.join(self.path_to_data_dir, "labels.csv")

    def features_extraction(self):

        data = pd.read_csv(self.path_to_data, header=None, index_col=None)
        X = data.iloc[:, :-2]
        y = data.iloc[:, -1]  # array  of signal_shape
        signal_shape_array = data.iloc[:, -2]  # array  of signal_shape

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]

        # #     #------------------------features + window------------------

        self.alt = X.shape[0]
        self.traj_num = 17
        # nombre de dipole
        self.dp_n = int(X.shape[1]/(self.signal_shape*self.traj_num))

        print(" le nombre de trajectoires est {} ".format(self.traj_num))
        print(" le nombre de dipoles est {} ".format(self.dp_n))

        df_features = pd.DataFrame(X)
        df_labels = pd.DataFrame(y)

        if not os.path.exists(self.path_to_data_dir):
            os.mkdir(self.path_to_data_dir)

        df_features.to_csv(os.path.join(self.path_to_data_dir,
                           "features.csv"), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir,
                         "labels.csv"), header=False, index=False)

    def features_extraction_segment(self, segment_width):

        data = pd.read_csv(self.path_to_data, header=None, index_col=None)
        X = data.iloc[:, :-2]
        y = data.iloc[:, -1]  # array  of signal_shape
        signal_shape_array = data.iloc[:, -2]  # array  of signal_shape

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]

        # #     #------------------------features + window------------------

        self.alt = X.shape[0]
        print(self.alt)
        self.traj_num = 17
        # nombre de dipole
        self.dp_n = int(X.shape[1]/(self.signal_shape*self.traj_num))

        print(" le nombre de trajectoires est {} ".format(self.traj_num))
        print(" le nombre de dipoles est {} ".format(self.dp_n))
        print(" signal shape est {} ".format(self.signal_shape))

        X = X.reshape(self.alt, self.traj_num, self.dp_n, self.signal_shape)

        X_new = []
        y_new = []

        # labelisation
        for alt in range(X.shape[0]):
            for i in range(self.traj_num):
                for k in range(int(self.signal_shape/segment_width)):

                    Li = []
                    for j in range(self.dp_n):
                        Li.extend(
                            list(X[alt][i][j][segment_width*k:segment_width*(k+1)]))

                    if i in range(3, 15) and (75 <= k*segment_width <= 125):
                        y_new.append(y[alt])

                    else:
                        y_new.append(y[alt])
                    X_new.append(Li)

        X_new = np.array(X_new)

        print(X_new.shape)
        y_new = np.array(y_new)

        df_features = pd.DataFrame(X_new.reshape(X_new.shape[0], -1))
        df_labels = pd.DataFrame(y_new)

        if not os.path.exists(self.path_to_data_dir):
            os.mkdir(self.path_to_data_dir)

        df_features.to_csv(os.path.join(self.path_to_data_dir,
                           "features.csv"), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir,
                         "labels.csv"), header=False, index=False)

    def data_split(self):

        self.features = pd.read_csv(
            self.path_to_features, header=None, index_col=None)
        self.labels = pd.read_csv(
            self.path_to_labels, header=None, index_col=None)

        print("features shape   :", self.features.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.33, stratify=np.array(self.labels), shuffle=True, random_state=42)

        X_train = np.array(X_train, dtype=np.float64)
        
        y_train = np.array(y_train, dtype=np.float64).reshape(
            (X_train.shape[0], ))
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64).reshape(
            (X_test.shape[0], ))

        print("end")

       
    
        return X_train, y_train, X_test, y_test



if __name__ == '__main__':

    artificial_intelligence = Artificial_intelligence()

    # artificial_intelligence.features_extraction()
    artificial_intelligence.features_extraction_segment(
        segment_width=10)  # generate features and labels

    X_train, y_train,  X_test, y_test = artificial_intelligence.data_split()  # use ExT2P1 data


    LR = False
    BR = False
    DT = False
    RF = False
    SVR_ = False
    KNR = False
    GB = False
    ET = True
    Lo = False
    Ri = False

    choice = [LR, BR, DT, RF, SVR_, KNR, GB, ET, Lo, Ri]
    
    dict_model = {"LR": LinearRegression(), "BR": BayesianRidge(),"DT": DecisionTreeRegressor(), 
                    "RF": RandomForestRegressor(), "SVR": SVR(), "KNR" : KNeighborsRegressor() ,
                    "GB": GradientBoostingRegressor(), "ET": ExtraTreesRegressor() , "Lo": Lasso() , "Ri":   Ridge()}
    
    name, model = list(dict_model.items())[choice.index(True)]

    print("----------- {} -----------------".format(name))

    model.fit(X_train, y_train)

    imp = []
 
    # dipole = [13, 16, 36, 18, 38, 68, 26, 28, 12, 27, 37, 45, 58]
    # dipole = [13, 18, 38, 28, 12, 27, 37, 45, 58]
    # dipole = [38, 27, 37, 45, 58]
    # dipole = [13, 16, 36, 18, 38, 68, 26, 28, 27, 37, 45,58]
    # dipole = [13, 18, 38, 12 , 27, 58]
    # dipole = [12]
    dipole = [12, 13, 16, 18, 26, 27, 28, 36, 37, 38, 45, 58, 68]
    
    # importance = abs(model.coef_)
    importance = model.feature_importances_
    # summarize feature importance
    # for i,v in enumerate(importance):
        # print('Feature: %0d, Score: %.5f' % (i,v))
        
    for i in range(0, len(importance), 10):
        v = 0
        for j in range(0,10):
            v = v + importance[i+j]
        print('Feature: %0d, Score: %.5f' % (i,v))   
        
        imp.append(v)   
        
    features = [  ]
    for i in dipole :
        features.append(f"dp{i}")
          
    # plot feature importance
 
    dict_feature1 = dict(zip(features, imp))
    dict_feature2 = OrderedDict(sorted(dict_feature1.items(), key=lambda t: t[1]))
    
    # plt.bar(dict_feature2.keys(), dict_feature2.values())
    
    
    zipped = list(zip(dict_feature2.keys(), dict_feature2.values()))
    df = pd.DataFrame(zipped, columns=['features', 'importances'])
    
    sns.barplot(x='features', y='importances', data=df)

    # summarize feature importance
    mse_train = mean_squared_error(model.predict(X_train), y_train)
    mse_test = mean_squared_error(model.predict(X_test), y_test)

    mae_train = mean_absolute_error(model.predict(X_train), y_train)
    mae_test = mean_absolute_error(model.predict(X_test), y_test)

    r_carre = model.score(X_test, y_test)
    print(" R carre  {}".format(r_carre))
    print()
    print("mse_train {}, mse_test {}, mae_train {}, mae_test {} ".format(
        mse_train, mse_test, mae_train, mae_test))

    plt.title( f"nombre de dipoles {len(dipole)} ; Modèle : {name} \n r_carre: {r_carre:.3f} mse_train: {mse_train:.3f}, mse_test: {mse_test:.3f}, mae_train {mae_train:.3f}, mae_test: {mae_test:.3f}")
    y_pred_lr = model.predict(X_test)


    plt.show()

    #########################################################################
    # %%
