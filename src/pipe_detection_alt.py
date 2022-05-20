
# %%

import os
import pickle
from cProfile import label
from tkinter.tix import Tree
from tracemalloc import Snapshot
from turtle import distance
from xmlrpc.client import TRANSPORT_ERROR


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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

import utils.augmentation as aug
import utils.helper as hlp
from altitude_manage import Data_extraction
import scipy.ndimage.filters as filters
import seaborn as sns

# ______________sckitlearn

# --------------- utils
class Artificial_intelligence(Data_extraction):

    def __init__(self, ESSAI=2, TEST=2, PIPE=1) -> None:
        super().__init__(ESSAI=ESSAI, TEST=TEST, PIPE=PIPE)
        # self.path_to_data = os.path.join(
        # self.path_to_data_dir, 'data_all_z_pipe_{}_no_pipe.csv'.format(self.pipe))
        # self.path_to_data = os.path.join(
        #     self.path_to_data_dir, 'data_all_z_pipe_{}.csv'.format(self.pipe))
        self.path_to_data = os.path.join(self.path_to_data_dir,  "data_T{}_P_{}_DP13_{}.csv".format(
            self.TEST, self.pipe, self.index_traj))
        self.path_to_features = os.path.join(
            self.path_to_data_dir, "features_T{}_P{}_{}.csv".format(self.TEST, self.pipe, self.index_traj))
        self.path_to_labels = os.path.join(self.path_to_data_dir, "labels_T{}_P{}_{}.csv".format(
            self.TEST, self.pipe, self.index_traj))
        self.path_to_models = os.path.join(os.getcwd(), "model")
        # self.path_to_alt = os.path.join(
        #     self.path_to_data_dir, "alt_no_pipe.csv")
        # self.path_to_alt = os.path.join(
        #     self.path_to_data_dir, "alt.csv")

        self.path_to_alt = os.path.join(self.path_to_data_dir, "alt_T{}_P{}_{}.csv".format(
            self.TEST, self.pipe, self.index_traj))

        sns.set()

    def features_extraction_segment_sliding(self, segment_width=10, split=False):

        seg_pipe = list(range(99, 300)) + list(range(649, 750))
        data = pd.read_csv(self.path_to_data, header=None, index_col=None)

        altitude = np.array(pd.read_csv(
            self.path_to_alt, header=None, index_col=None), dtype=np.float64)

        self.traj_num = altitude.shape[0]

        X = data.iloc[:, :-2]
        y = data.iloc[:, -1]  # array  of signal_shape
        signal_shape_array = data.iloc[:, -2]  # array  of signal_shape

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]

        # #     #------------------------features + window------------------

        self.alt = X.shape[0]
        # nombre de dipole

        self.dp_n = int(X.shape[1]/(self.signal_shape*self.traj_num))

        X = X.reshape(self.alt, self.traj_num, self.dp_n, self.signal_shape)

        X_new = []
        y_new = []

        X_new_pipe = []
        y_new_pipe = []

        # labelisation
        for alt in range(X.shape[0]):
            for i in range(self.traj_num):
                for k in range(int(self.signal_shape-segment_width)):
                    Li = []
                    for j in range(self.dp_n):
                        Li.extend(
                            list(X[alt][i][j][k:k+segment_width]))

                    if (k in seg_pipe) & split:
                        X_new_pipe.append(Li)
                        y_new_pipe.append(
                            np.mean(altitude[i, k:k+segment_width]))
                    else:
                        y_new.append(np.mean(altitude[i, k:k+segment_width]))
                        X_new.append(Li)

        X_new = np.array(X_new)
        X_new_pipe = np.array(X_new_pipe)
        y_new = np.array(y_new)
        y_new_pipe = np.array(y_new_pipe)

        print("X in features split ", X_new.shape)
        print("Y in features split ", y_new.shape)

        print("alt min ", np.min(y_new))
        print("alt max ", np.max(y_new))

        df_features = pd.DataFrame(X_new.reshape(X_new.shape[0], -1))
        df_labels = pd.DataFrame(y_new)

        if not os.path.exists(self.path_to_data_dir):
            os.mkdir(self.path_to_data_dir)

        df_features.to_csv(os.path.join(self.path_to_data_dir,
                           "features_T{}_P{}_{}.csv".format(self.TEST, self.pipe, self.index_traj)), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir,
                         "labels_T{}_P{}_{}.csv".format(self.TEST, self.pipe, self.index_traj)), header=False, index=False)

        return X_new_pipe, y_new_pipe

    def features_extraction_segment(self, segment_width):

        data = pd.read_csv(self.path_to_data, header=None, index_col=None)

        altitude = np.array(pd.read_csv(
            self.path_to_alt, header=None, index_col=None), dtype=np.float64)

        seg_keep = [7, 8, 9, 10, 11, 12, 13]

        self.traj_num = altitude.shape[0]

        X = data.iloc[:, :-2]
        y = data.iloc[:, -1]  # array  of signal_shape
        signal_shape_array = data.iloc[:, -2]  # array  of signal_shape

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]

        # #     #------------------------features + window------------------

        self.alt = X.shape[0]
        # nombre de dipole

        self.dp_n = int(X.shape[1]/(self.signal_shape*self.traj_num))

        print(" le nombre de trajectoires est {} ".format(self.traj_num))
        print(" le nombre de dipoles est {} ".format(self.dp_n))
        print(" signal shape est {} ".format(self.signal_shape))

        X = X.reshape(self.alt, self.traj_num, self.dp_n, self.signal_shape)

        X_new = []
        X_new1 = []
        y_new = []
        y_new1 = []

        # labelisation
        for alt in range(X.shape[0]):
            for i in range(self.traj_num):
                for k in range(int(self.signal_shape/segment_width)):
                    Li = []
                    for j in range(self.dp_n):
                        Li.extend(
                            list(X[alt][i][j][segment_width*k:segment_width*(k+1)]))

                    if k in seg_keep:
                        y_new.append(
                            np.mean(altitude[i, segment_width*k:segment_width*(k+1)]))
                        X_new.append(Li)
                    else:
                        y_new.append(
                            np.mean(altitude[i, segment_width*k:segment_width*(k+1)]))
                        X_new.append(Li)

        X_new = np.array(X_new)

        print("X in features split ", X_new.shape)
        y_new = np.array(y_new)

        df_features = pd.DataFrame(X_new.reshape(X_new.shape[0], -1))
        df_labels = pd.DataFrame(y_new)

        if not os.path.exists(self.path_to_data_dir):
            os.mkdir(self.path_to_data_dir)

        df_features.to_csv(os.path.join(self.path_to_data_dir,
                           "features_T{}_P{}_{}.csv".format(self.TEST, self.pipe, self.index_traj)), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir,
                         "labels_T{}_P{}_{}.csv".format(self.TEST, self.pipe, self.index_traj)), header=False, index=False)

        X_new1 = np.array(X_new1)
        y_new1 = np.array(y_new1)

        print("alt min ", np.min(y_new))
        print("alt max ", np.max(y_new))

        return X_new1, y_new1

    def data_split(self, merge_data=False, split=True):

        if merge_data:
            self.features = pd.read_csv(os.path.join(
                self.path_to_data_dir, "features_merge.csv"), header=None, index_col=None)
            self.labels = pd.read_csv(os.path.join(
                self.path_to_data_dir, "labels_merge.csv"), header=None, index_col=None)
        else:
            self.features = pd.read_csv(
                self.path_to_features, header=None, index_col=None)
            self.labels = pd.read_csv(
                self.path_to_labels, header=None, index_col=None)

        print("features shape   : ", self.features.shape)
        print("labels shape   : ", self.labels.shape)

        if split:
            # X_train, X_test, y_train, y_test, index_x, index_y = train_test_split(
            #     self.features, self.labels, range(self.labels.shape[0]), test_size=0.33,
            #     stratify=np.array(self.labels), shuffle=True, random_state=42)
            X_train, X_test, y_train, y_test, index_x, index_y = train_test_split(
                self.features, self.labels, range(self.labels.shape[0]), test_size=0.33, shuffle=True, random_state=42)

            X_train = np.array(X_train, dtype=np.float64)
            y_train = np.array(y_train, dtype=np.float64).reshape(
                (X_train.shape[0], ))
            X_test = np.array(X_test, dtype=np.float64)
            y_test = np.array(y_test, dtype=np.float64).reshape(
                (X_test.shape[0], ))
        else:

            X_train, y_train, index_y = shuffle(
                self.features, self.labels, range(self.labels.shape[0]))  # shuffle data

            X_train = np.array(X_train, dtype=np.float64)
            y_train = np.array(y_train, dtype=np.float64).reshape(
                (X_train.shape[0], ))
            X_test = np.ones(shape=X_train.shape)
            y_test = np.ones(shape=y_train.shape)

        print("features_train {} features_test {}".format(
            X_train.shape, X_test.shape))

        return X_train, y_train, X_test, y_test, index_y

    def association(self, test_list, pipe_list, index_traj_list):

        path_feature_ini = os.path.join(self.path_to_data_dir, "features_T{}_P{}_{}.csv".format(
            test_list[0], pipe_list[0], index_traj_list[0]))
        path_label_init = os.path.join(self.path_to_data_dir, "labels_T{}_P{}_{}.csv".format(
            test_list[0], pipe_list[0], index_traj_list[0]))
        X = np.array(pd.read_csv(path_feature_ini,
                     header=None, index_col=None))
        y = np.array(pd.read_csv(path_label_init, header=None, index_col=None))

        for t, p, i in list(zip(test_list[1:], pipe_list[1:], index_traj_list[1:])):

            path_feature = os.path.join(
                self.path_to_data_dir, "features_T{}_P{}_{}.csv".format(t, p, i))
            path_label = os.path.join(
                self.path_to_data_dir, "labels_T{}_P{}_{}.csv".format(t, p, i))
            X = np.concatenate(
                (X, np.array(pd.read_csv(path_feature, header=None, index_col=None))))
            y = np.concatenate(
                (y, np.array(pd.read_csv(path_label, header=None, index_col=None))))

        df_features = pd.DataFrame(X)
        df_labels = pd.DataFrame(y)
        df_features.to_csv(os.path.join(self.path_to_data_dir,
                           "features_merge.csv"), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir,
                         "labels_merge.csv"), header=False, index=False)

    def load_E2T2P2_data(self, segment_width=10):

        path = os.path.join(self.path_to_data_dir,
                            'data_all_z_pipe_1_pipe.csv')
        path_alt = os.path.join(self.path_to_data_dir, 'alt_pipe.csv')

        data = pd.read_csv(path, header=None, index_col=None)

        altitude = np.array(pd.read_csv(
            path_alt, header=None, index_col=None), dtype=np.float64)

        self.traj_num = altitude.shape[0]

        altitude = altitude.ravel().reshape(-1, 10)

        X = data.iloc[:, :-2]
        y = data.iloc[:, -1]  # array  of signal_shape
        signal_shape_array = data.iloc[:, -2]  # array  of signal_shape

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]

        # #     #------------------------features + window------------------

        self.alt = X.shape[0]

        # nombre de dipole
        self.dp_n = int(X.shape[1]/(self.signal_shape*self.traj_num))

        print(" le nombre de trajectoires est {} ".format(self.traj_num))
        print(" le nombre de dipoles est {} ".format(self.dp_n))
        print(" signal shape est {} ".format(self.signal_shape))

        X = X.reshape(self.alt, self.traj_num, self.dp_n, self.signal_shape)

        X_new = []
        y_new = []
        X_new1 = []
        y_new1 = []
        # labelisation
        for alt in range(X.shape[0]):
            for i in range(self.traj_num):
                for k in range(int(self.signal_shape/segment_width)):

                    Li = []
                    for j in range(self.dp_n):
                        Li.extend(
                            list(X[alt][i][j][segment_width*k:segment_width*(k+1)]))

                        if k in [8, 9, 10, 11, 12]:
                            y_new1.append(np.mean(altitude[i, :]))
                            X_new1.append(Li)
                        else:
                            y_new.append(np.mean(altitude[i, :]))
                            X_new.append(Li)

        X_new = np.array(X_new)
        y_new = np.array(y_new)
        X_new1 = np.array(X_new1)
        y_new1 = np.array(y_new1)

        # X_new1 = np.random.rand(3315,130).reshape(3315,130)

        return X_new1, y_new1

    def traj_for_test(self, i=6, segment_width=10):

        path = os.path.join(self.path_to_data_dir,
                            'data_traj_i/data_T1_P_1_DP13_traj_{}.csv'.format(i))
        path_alt = os.path.join(self.path_to_data_dir,
                                'data_traj_i/alt_T1_P1_traj_{}.csv'.format(i))

        data = pd.read_csv(path, header=None, index_col=None)

        altitude = np.array(pd.read_csv(
            path_alt, header=None, index_col=None), dtype=np.float64)

        self.traj_num = altitude.shape[0]

        X = data.iloc[:, :-2]
        y = data.iloc[:, -1]  # array  of signal_shape
        signal_shape_array = data.iloc[:, -2]  # array  of signal_shape

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]

        # #     #------------------------features + window------------------

        self.alt = X.shape[0]

        # nombre de dipole
        self.dp_n = int(X.shape[1]/(self.signal_shape*self.traj_num))

        print(" le nombre de trajectoires est {} ".format(self.traj_num))
        print(" le nombre de dipoles est {} ".format(self.dp_n))
        print(" signal shape est {} ".format(self.signal_shape))

        X = X.reshape(self.alt, self.traj_num, self.dp_n, self.signal_shape)

        for i in range(13):

            plt.plot(X[0][0][i][:])

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

                    X_new.append(Li)
                    y_new.append(
                        np.mean(altitude[i, segment_width*k:segment_width*(k+1)]))

        X_new = np.array(X_new)
        y_new = np.array(y_new)

        # X_new1 = np.random.rand(3315,130).reshape(3315,130)

        return X_new, y_new


# %%
if __name__ == '__main__':

    artificial_intelligence = Artificial_intelligence(ESSAI=2, TEST=1, PIPE=1)


# %%
    X_test1, y_test1 = artificial_intelligence.features_extraction_segment_sliding(
        segment_width=10, split=False)

    # X_test_,  y_test_ = artificial_intelligence.features_extraction_segment(
    #     segment_width=10)  # generate features and labels
# %%
    artificial_intelligence.association(test_list=[1, 2, 2, 2], pipe_list=[
                                        1, 1, 2, 3], index_traj_list=["no_pipe", "no_pipe", "no_pipe", "no_pipe"])
# %%

    X_train, y_train,  X_test2, y_test2, indice_test = artificial_intelligence.data_split(
        merge_data=False, split=False)  # use ExT2P1 data

# %%
    i = 11
    X_test, y_test = artificial_intelligence.traj_for_test(i=i)
    # X_test , y_test = artificial_intelligence.load_E2T2P2_data(segment_width=10)
    # X_train , y_train, indice_train = shuffle(X_train , y_train, range(y_train.shape[0])) # shuffle data
    # X_test , y_test, indice_test = shuffle(X_test , y_test, range(y_test.shape[0])) # shuffle data

    # X_test, y_test = X_test1, y_test1
    # X_test, y_test = X_test2, y_test2
# %%
    #" visualization repartition"

    # fig, ax = plt.subplots(1,2)

    # ax = ax.flatten()
    # ax[0].hist(y_train)
    # ax[0].set_title('alt_train',
    #          fontweight ="bold")
    # ax[1].hist(y_test) # repartition
    # ax[1].set_title('alt_test',
    #         fontweight ="bold")
    # fig.suptitle('Repartition des altitudes T1 & fenetre fixe')
    # plt.show()

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

    dict_model = {"LR": LinearRegression(), "BR": BayesianRidge(), "DT": DecisionTreeRegressor(),
                  "RF": RandomForestRegressor(), "SVR": SVR(), "KNR": KNeighborsRegressor(),
                  "GB": GradientBoostingRegressor(), "ET": ExtraTreesRegressor(), "Lo": Lasso(), "Ri":   Ridge()}

    name, model = list(dict_model.items())[choice.index(True)]

    print("----------- {} -----------------".format(name))
    # # #------------------- hyperparameters -----------------
    # artificial_intelligence.find_best_learning_params(model, type=name)

    # model.set_params(**artificial_intelligence.best_params)

    # model = LocalOutlierFactor(novelty=True, n_neighbors=25, contamination=0.0001)
    # model = OneClassSVM(gamma='auto')
    # model.fit(X_train)
    # y_pred = model.predict(X_test)

    # plt.scatter(np.linspace(40, 460,y_pred.shape[0]), y_pred, s=2, label=f"{i}")
    # plt.figure()
    # plt.plot(model.score_samples(X_test), label=f"{i}")
    # plt.legend()


# %%
    model.fit(X_train, y_train)

    # rcarre_list = []
    # for cpt in range(1,14):
    #     X_test, y_test = artificial_intelligence.traj_for_test(i=cpt)
    #     rcarre = model.score(X_test, y_test)
    #     if rcarre ==1.0 :
    #         rcarre = 0.91
    #     rcarre_list.append(rcarre)

    # plt.plot(range(1,14), rcarre_list)
    # plt.xlabel("trajectoire")
    # plt.ylabel("rcarre")
    # plt.title("ExtraTrees")

    hlp.evaluate_model(model, X_train, y_train, X_test, y_test)

    # section = [  ]
    # tra = []
    # alt = []
    # for i in indice_test :
    #     section.append((i%30)+1)
    #     tra.append(int(i/30)  %7 +1)
    #     alt.append(int(i/(30*7) ) + 1)

    # section = np.array(section).reshape(-1,1)
    # # tra = np.array(tra).reshape(-1,1)
    # alt = np.array(alt).reshape(-1,1)

    y_pred = model.predict(X_test)

    # error = abs(y_test.reshape(-1, 1)-y_pred.reshape(-1, 1))
    error = (y_pred.reshape(-1, 1)-y_test.reshape(-1, 1))/y_test.reshape(-1, 1)

    # i = (section == 8) | (section == 10)

    # print(error[i])

    # column = ["id", "alt", "sec", "tj", "y_test", "y_pred", "error" ]
    # df = pd.DataFrame(np.concatenate((np.array(indice_test).reshape(-1,1), alt, section, tra, y_test.reshape(-1, 1), y_pred.reshape(-1, 1),
    #   error), axis=1), columns=column)

    df = pd.DataFrame(np.concatenate(
        (y_test.reshape(-1, 1), y_pred.reshape(-1, 1), error), axis=1))

    df.to_csv(os.path.join(artificial_intelligence.path_to_data_dir,
              'error.csv'), header=False, index=False)

    print(np.max(abs(y_test.reshape(-1, 1)-y_pred.reshape(-1, 1))))

 # %%

    plt.figure()
    plt.plot(np.linspace(40, 460, error.shape[0]), y_test, label='alt_true')
    plt.plot(np.linspace(40, 460, error.shape[0]), y_pred, label='alt_pred')

    plt.figure()

    plt.plot(np.linspace(40, 460, error.shape[0]), hlp.standardization(
        y_test), label='alt_true')
    plt.plot(np.linspace(40, 460, error.shape[0]), hlp.standardization(
        y_pred), label='alt_pred')
    plt.xlabel("distance (cm) selon x")
    plt.ylabel("alttitude")
    plt.title(
        "evolution de prediction( traj :{}) en fonction de la position, modèle :{} ".format(i, name))
    plt.legend()
    plt.figure()

    plt.plot(np.linspace(40, 460, error.shape[0]), error)
    plt.title(
        "evolution de prediction( traj :{}) en fonction de la position,  modèle :{} ".format(i, name))
    plt.xlabel("distance (cm) selon x")
    plt.ylabel("erreur relatif")
    plt.legend()
    # plt.plot(np.arange(error.shape[0])*4.33, filters.gaussian_filter1d(error, 10))

# %%
