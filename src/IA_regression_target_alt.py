# %%


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
# --------------- utils

import utils.augmentation as aug
import utils.helper as hlp




class Artificial_intelligence(Data_extraction):

    def __init__(self) -> None:
        super().__init__(ESSAI=2, TEST=2)
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
        

        X_train, X_test, y_train, y_test = train_test_split( self.features , self.labels, test_size=0.33, stratify= np.array(self.labels), shuffle = True, random_state=42)
        

        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64).reshape(
            (X_train.shape[0], ))
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64).reshape(
            (X_test.shape[0], ))
        print("end")

        return X_train, y_train, X_test, y_test

    def find_best_learning_params(self, model, type='lr'):

        X_train, y_train, _, _ = self.data_split()

        if type == 'lr':
            hyper_param_grid = {
                'solver': ['svd', 'cholesky', 'lsqr', 'sag'],

                'fit_intercept': [True, False],

            }

        else:
            print("model type incorrect")
        # ---------------------------------------

        cv = GridSearchCV(model, param_grid=hyper_param_grid,
                          cv=7, n_jobs=-1, verbose=0)

        cv.fit(X_train, y_train)

        self.best_params = cv.best_params_
        print(self.best_params)
    

    def load_E2T2P2_data(self, segment_width=10):
        
        
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
        y_new = np.array(y_new)
        
        return X_new, y_new
        

if __name__ == '__main__':

    artificial_intelligence = Artificial_intelligence()

    # artificial_intelligence.features_extraction() 
    artificial_intelligence.features_extraction_segment(segment_width=10) # generate features and labels 

    X_train, y_train ,  X_test , y_test= artificial_intelligence.data_split() # use ExT2P1 data

    LR = False
    BR = False
    DT = False
    RF = True
    SVR_ = False

    choice = [LR, BR, DT, RF, SVR_]

    model1 = LinearRegression()
    model2 = BayesianRidge()
    model3 = DecisionTreeRegressor()
    model4 = RandomForestRegressor()
    model5 = SVR()

    dict_model = {"LR": model1, "BR": model2,
                  "DT": model3, "RF": model4, "SVR": model5}
    name, model = list(dict_model.items())[choice.index(True)]

    print("----------- {} -----------------".format(name))

    model.fit(X_train, y_train)
# %%

    # X_test , y_test = artificial_intelligence.load_E2T2P2_data(segment_width=10)
    # X_test , y_test, indice_test = shuffle(X_test , y_test, range(y_test.shape[0])) # shuffle data
    
    
    mse_train = mean_squared_error(model.predict(X_train), y_train)
    mse_test = mean_squared_error(model.predict(X_test), y_test)

    mae_train = mean_absolute_error(model.predict(X_train), y_train)
    mae_test = mean_absolute_error(model.predict(X_test), y_test)

    print(" R carre  {}".format(model.score(X_test, y_test)))
    print()

    print("mse_train {}, mse_test {}, mae_train {}, mae_test{} ".format(mse_train, mse_test, mae_train, mae_test))
    

    y_pred_lr = model.predict(X_test)
    
    df = pd.DataFrame(np.concatenate((y_test.reshape(-1,1), y_pred_lr.reshape(-1,1), abs(y_test.reshape(-1,1)-y_pred_lr.reshape(-1,1))), axis=1))

    df.to_csv(os.path.join(artificial_intelligence.path_to_data_dir, 'test.csv'), header=False, index=False)


    #########################################################################
    # # #-------------------hyperparameters -----------------
    # artificial_intelligence.find_best_learning_params(clf, type=type)
    # clf.set_params(**artificial_intelligence.best_params)
    # # #-------------------

    indices = [ ]
    section = [ ]
    pas = 0.5
    alt_z_val = np.arange(4, 12+pas, pas)
    
    indice_test = range(X_test.shape[0])
    for i in indice_test :
        alt = int(i/(29*17)) # altitude
        traj = int(i/29) % 17
        
        sec = i%29
        indices.append(int(traj +1))
        section.append(sec)
        
    indices = np.array(indices).reshape(-1,1)
    section = np.array(section).reshape(-1,1)
    
   # %% 
   # visualisation des erreurs sous forme de segments

   
    df = pd.DataFrame(np.concatenate((y_test.reshape(-1,1), y_pred_lr.reshape(-1,1), abs(y_test.reshape(-1,1)-y_pred_lr.reshape(-1,1)),  indices, section), axis=1))
    df.to_csv(os.path.join(artificial_intelligence.path_to_data_dir, 'test.csv'), header=False, index=False)

    w1 = 80
    w2 = 150
    lines = [[(i, y_test[i]), (i, y_pred_lr[i])] for i in range(w1,w2)]

    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.scatter(range(w1,w2), y_pred_lr[w1:w2], label= "pred", s=4, c='red')
    ax.scatter(range(w1,w2), y_test[w1:w2], label= "label", s=4, c='black')
    ax.autoscale()
    ax.margins(0.1)
    plt.title(" comparaison y_true et y_pred")
    plt.xlabel(" echantillons ")
    plt.ylabel(" alt(cm) ")
    plt.legend()
    plt.show()
    
# %% PLot the error
    w1 = w1
    w2 = w2
    plt.figure()
    plt.plot(np.abs(y_pred_lr-y_test)[w1:w2])
    plt.title("valeur absolue de l'erreur")
    plt.xlabel(" echantillons ")
    plt.ylabel(" abs(ytest_ypred) ")
    plt.legend()
    plt.show()

# %%
