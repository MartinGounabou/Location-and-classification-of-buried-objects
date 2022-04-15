
# %%

from cgi import test
from operator import index
from statistics import mode
from data_manipulation_burried_object_localisation import Data_extraction

import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
import pylab as pl
from matplotlib import collections as mc

# ______________sckitlearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer, accuracy_score, precision_score, mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
# --------------- utils

import utils.augmentation as aug
import utils.helper as hlp

# %%


class Artificial_intelligence(Data_extraction):

    def __init__(self) -> None:
        super().__init__(ESSAI=1, TEST=2)
        self.path_to_data = os.path.join(
            self.path_to_data_dir, 'data_all_z.csv')
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

        
        self.features =  pd.read_csv(self.path_to_features, header=None, index_col=None)
        self.labels = pd.read_csv(self.path_to_labels, header=None, index_col=None)
        
        print("features shape   :" , self.features.shape)
        

        X_train, X_test, y_train, y_test, indice_train, indice_test = train_test_split( self.features , self.labels, range(self.labels.shape[0]),  test_size=0.33, stratify= np.array(self.labels), shuffle = True, random_state=42)
        
        print("test indice ", indice_test)
        
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64).reshape(
            (X_train.shape[0], ))
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64).reshape(
            (X_test.shape[0], ))
        print("end")

        return X_train, y_train, X_test, y_test


    def test(self, segment_width=10):
        
        
        path = os.path.join(self.path_to_data_dir, 'data_all_z_pipe2.csv')
        
        data = pd.read_csv(path, header=None, index_col=None)
        
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
        
        return X_new, y_new
        

# %%
if __name__ == '__main__':

    artificial_intelligence = Artificial_intelligence()

    # artificial_intelligence.features_extraction() 
    artificial_intelligence.features_extraction_segment(segment_width=10) 


    X_train, y_train, _ , _ = artificial_intelligence.data_split()


    # X_test , y_test = artificial_intelligence.test(segment_width=10)
    
    
    # # X_test , y_test = shuffle(X_test[:10000] , y_test[:10000])

    # param_grid = {  'bootstrap': [True], 'max_depth': [5, 10, None],
    #     'max_features': ['auto', 'log2'], 
    #         'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}

    # model = RandomForestRegressor(random_state = 1)

    # g_search = GridSearchCV(estimator = model, param_grid = param_grid, 
    #  cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)
    
    # g_search.fit(X_train, y_train)

    # model.set_params(**g_search.best_params_)
    # # #-------------------

    # model.fit(X_train, y_train)

    # print(" prediction score : ", model.score(X_test, y_test))


    # y_pred_lr = model.predict(X_test)
    
    # df = pd.DataFrame(np.concatenate((y_test.reshape(-1,1), y_pred_lr.reshape(-1,1), abs(y_test.reshape(-1,1)-y_pred_lr.reshape(-1,1))), axis=1))

    # df.to_csv(os.path.join(artificial_intelligence.path_to_data_dir, 'test.csv'), header=False, index=False)
    
 
    # mse_train = mean_squared_error(model.predict(X_train), y_train)
    # mse_test = mean_squared_error(model.predict(X_test), y_test)

    # mae_train = mean_absolute_error(model.predict(X_train), y_train)
    # mae_test = mean_absolute_error(model.predict(X_test), y_test)

    # print(" R carre  {}".format(model.score(X_test, y_test)))
    # print()

    # print("mse_train {}, mse_test {}, mae_train {}, mae_test{} ".format(mse_train, mse_test, mae_train, mae_test))

# %%



    # parameters = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
    #             'learning_rate' : (0.05,0.25,0.50,1),
    #             'criterion' : ['friedman_mse', 'mse', 'mae'],
    #             'max_features' : ['auto', 'sqrt', 'log2']
    #             }

    # grid = GridSearchCV(GradientBoostingRegressor(),parameters)
    
    # model = grid.fit(X_train,y_train)
    # print(model.best_params_,'\n')
    # print(model.best_estimator_,'\n')

    # GradientBoostingRegressor(learning_rate=0.25, loss='lad', max_features='sqrt')    


    
    # print("----------- {} -----------------".format(name))

    # model.fit(X_train, y_train)

    # y_pred_lr = model.predict(X_test)
    
    # df = pd.DataFrame(np.concatenate((y_test.reshape(-1,1), y_pred_lr.reshape(-1,1)), axis=1))

    # df.to_csv(os.path.join(artificial_intelligence.path_to_data_dir, 'test.csv'), header=False, index=False)
    
 
    # mse_train = mean_squared_error(model.predict(X_train), y_train)
    # mse_test = mean_squared_error(model.predict(X_test), y_test)

    # mae_train = mean_absolute_error(model.predict(X_train), y_train)
    # mae_test = mean_absolute_error(model.predict(X_test), y_test)

    # print(" R carre  {}".format(model.score(X_test, y_test)))
    # print()

    # print("mse_train {}, mse_test {}, mae_train {}, mae_test{} ".format(mse_train, mse_test, mae_train, mae_test))

# head = 10
# for model in regressors[:head]:
#     start = time()
#     model.fit(X_train, y_train)
#     train_time = time() - start
#     start = time()
#     y_pred = model.predict(X_test)
#     predict_time = time()-start    
#     print(model)
#     print("\tTraining time: %0.3fs" % train_time)
#     print("\tPrediction time: %0.3fs" % predict_time)
#     print("\tExplained variance:", explained_variance_score(y_test, y_pred))
#     print("\tMean absolute error:", mean_absolute_error(y_test, y_pred))
#     print("\tR2 score:", r2_score(y_test, y_pred))
#     print()
    
# regressors = [
#     KNeighborsRegressor(),
#     GradientBoostingRegressor(),
#     KNeighborsRegressor(),
#     ExtraTreesRegressor(),
#     RandomForestRegressor(),
#     DecisionTreeRegressor(),
#     LinearRegression(),
#     Lasso(),
#     Ridge()
# ]

# from time import time

# from sklearn.linear_model import LinearRegression, Ridge,Lasso
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor

# from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score

# parameters = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
#               'learning_rate' : (0.05,0.25,0.50,1),
#               'criterion' : ['friedman_mse', 'mse', 'mae'],
#               'max_features' : ['auto', 'sqrt', 'log2']
#              }

# grid = GridSearchCV(GradientBoostingRegressor(),parameters)
# model = grid.fit(X_sc,y)
# print(model.best_params_,'\n')
# print(model.best_estimator_,'\n')

# {'criterion': 'friedman_mse', 'learning_rate': 0.25, 'loss': 'lad', 'max_features': 'sqrt'} 

# GradientBoostingRegressor(learning_rate=0.25, loss='lad', max_features='sqrt')    
    
    
    
    
    
    

    

    # y_pred_lr = model.predict(X_test)