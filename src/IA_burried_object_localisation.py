from email.headerregistry import HeaderRegistry
from genericpath import exists
<<<<<<< HEAD
from math import fabs
from operator import index
from this import d
from tkinter.tix import Tree

from sklearn.naive_bayes import GaussianNB
=======
from operator import index
>>>>>>> 2377358f6f5ed7feb032a74721058403d42f83b3
from data_manipulation_burried_object_localisation import Data_extraction

import numpy as np
import scipy as sp
import pandas as pd
import os 
import matplotlib.pyplot as plt
#%%

#______________sckitlearn
<<<<<<< HEAD
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
=======
from sklearn.model_selection import train_test_split

>>>>>>> 2377358f6f5ed7feb032a74721058403d42f83b3
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

<<<<<<< HEAD

from sklearn.ensemble import ExtraTreesClassifier
=======
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
>>>>>>> 2377358f6f5ed7feb032a74721058403d42f83b3
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
<<<<<<< HEAD
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer, accuracy_score, precision_score



import utils.augmentation as aug
import utils.helper as hlp

=======
>>>>>>> 2377358f6f5ed7feb032a74721058403d42f83b3
#%%

class Artificial_intelligence(Data_extraction) :
    
    def __init__(self) -> None:
        super().__init__()
<<<<<<< HEAD
        self.path_to_data = os.path.join(self.path_to_data_dir, 'data.csv')
        self.path_to_features = os.path.join(self.path_to_data_dir, "features.csv")
        self.path_to_labels = os.path.join(self.path_to_data_dir, "labels.csv")
        
    
    def features_extraction(self,window_h=1):
            
        data = pd.read_csv(self.path_to_data, header=None, index_col=None)
        X = data.iloc[:,:-2]
        y = data.iloc[:,-1] # array  of signal_shape
        signal_shape_array = data.iloc[:,-2] # array  of signal_shape
        
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]
    
        # #     #------------------------features + window------------------


        self.dp_n = int(X.shape[1]/ self.signal_shape) # nombre de dipole
        self.traj_num = X.shape[0]
        
        print(" le nombre de trajectoires est {} ".format(self.traj_num))    
        print(" le nombre de dipoles est {} ".format(self.dp_n))  
        
        
        N = int(self.signal_shape/window_h)
        X_window = np.zeros((self.traj_num,N*self.dp_n))
            
        for i  in range(X.shape[0]) :
        
            intensity_traj_i = X[i,:].reshape(13, self.signal_shape)
            intensity_traj_i_window =  np.zeros((13,N))
        
            for j in range(self.dp_n) :
                T = np.zeros(N)
                for cpt in range(N) :
                    
                    T[cpt] = np.sum(intensity_traj_i[j, cpt:cpt+window_h])

                intensity_traj_i_window[j] = T
                
            X_window[i] = intensity_traj_i_window.ravel()
            
        df_features = pd.DataFrame(X_window)
        df_labels = pd.DataFrame(y)
         
        if not os.path.exists(self.path_to_data_dir) :
            os.mkdir(self.path_to_data_dir)
            
        df_features.to_csv(os.path.join(self.path_to_data_dir, "features.csv"), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir, "labels.csv"), header=False, index=False)
        
        self.features = df_features
        self.labels = df_labels
     
    
    def features_extraction_segment(self,segment_width):
        
        # donnees diople, taille signal, label   
        data = pd.read_csv(self.path_to_data, header=None, index_col=None)
        X = data.iloc[:,:-2]
        y = data.iloc[:,-1] # array  of signal_shape
        signal_shape_array = data.iloc[:,-2] # array  of signal_shape
        
        
        # Array type
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0] # = 300, signal complet 
    
        # #     #------------------------features + window------------------

        self.dp_n = int(X.shape[1]/ self.signal_shape) # nombre de dipole
        self.traj_num = X.shape[0]
        
        print(" le nombre de trajectoires est {} ".format(self.traj_num))    
        print(" le nombre de dipoles est {} ".format(self.dp_n))    
        print(" signal shape est {} ".format(self.signal_shape))    
        
        X = X.reshape(self.traj_num,self.dp_n, self.signal_shape)  
        
        X_new = []
        y = []
        
        # labelisation 
        for i in range(17) :
            for k in range(int(self.signal_shape/segment_width)):
                Li = []
                for j in range(self.dp_n) :
                    Li.append(list(X[i][j][segment_width*k:segment_width*(k+1)]))
                
                if i in range(3,15) and ( 75 <= k*segment_width <= 125 ):
                    y.append(1)
        
                else :
                    y.append(0)
                X_new.append(Li)
    
        X_new = np.array(X_new)
        y = np.array(y)
        
        df_features = pd.DataFrame(X_new.reshape(X_new.shape[0],-1))
        df_labels = pd.DataFrame(y)
        
        if not os.path.exists(self.path_to_data_dir) :
            os.mkdir(self.path_to_data_dir)
            
        df_features.to_csv(os.path.join(self.path_to_data_dir, "features.csv"), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir, "labels.csv"), header=False, index=False)
        
        self.features = df_features
        self.labels = df_labels
        self.signal_shape = segment_width # 10
        
        
    def data_augmentation(self):
        
        X = np.array(self.features, dtype=np.float64)
        y = np.array(self.labels, dtype=np.float64)
        
        X_jiffer = X.copy()
        X_scaling = X.copy()
        X_permutation = X.copy()
        X_magnitude_warp = X.copy()
        X_rotation = X.copy()
        X_window_slice = X.copy()
        X_spawner= X.copy()
        X_wdba= X.copy()
        X_random_guided_warp = X.copy()
        X_discriminative_guided_warp = X.copy()
        
        
        for i in range(X.shape[0]):
            
            dp_values = X[i]            
            dp_values = dp_values.reshape((self.dp_n, self.signal_shape, 1))
            dp_jiffer = aug.jitter(dp_values, sigma=1e-5)
            dp_scaling = aug.scaling(dp_values, sigma=0.05)
            dp_permutaion = aug.permutation(dp_values)
            dp_magnitude_warp = aug.magnitude_warp(dp_values)
            dp_rotation = aug.rotation(dp_values)
            dp_window_slice = aug.window_slice(dp_values)
            print("\n ------------------spawner launch--------------------\n")
            dp_spawner = aug.spawner(dp_values, np.array([0]*13), sigma=4*1e-15)
            print("end")
            print("\n ------------------wdba launch--------------------\n")
            dp_wdba = aug.wdba(dp_values, np.array([0]*13), verbose=-1)
            print("\n ------------------random_guided_warp launch--------------------\n")
            dp_random_guided_warp = aug.random_guided_warp(dp_values, np.array([0]*13), verbose=-1)
            print("\n ------------------discriminative_guided_warp launch--------------------\n")
            dp_discriminative_guided_warp = aug.discriminative_guided_warp(dp_values, np.array([0]*13), verbose=-1)
            
            
            X_jiffer[i] = dp_jiffer.flatten()
            X_scaling[i] = dp_scaling.flatten()
            X_permutation[i] = dp_permutaion.flatten()
            X_magnitude_warp[i] = dp_magnitude_warp.flatten()
            X_rotation[i] = dp_rotation.flatten()
            X_window_slice[i] = dp_window_slice.flatten()
            X_spawner[i] = dp_spawner.flatten()
            X_wdba[i] = dp_wdba.flatten()
            X_random_guided_warp[i] = dp_random_guided_warp.flatten()
            X_discriminative_guided_warp[i] = dp_discriminative_guided_warp.flatten()

            # plt.plot(X[i,:self.signal_shape])
            # plt.plot(dp_discriminative_guided_warp[:self.signal_shape][0])
            # plt.plot(X[i,:10])
            # plt.plot(dp_discriminative_guided_warp[:10][0])
            # plt.show()
        
        # for i in range(12):
        #     plt.figure()
        #     plt.plot(X_window_slice[i, :])
        # plt.show()
            
        X = np.concatenate((X, X_jiffer,X_scaling, X_permutation, 
                            X_magnitude_warp, X_rotation, X_window_slice, 
                            X_spawner, X_wdba, X_random_guided_warp,
                            X_discriminative_guided_warp), axis=0)
        
        print("features :", X.shape)
        
        y = np.array(list(y.flatten())*11).reshape(-1,1)


        df_features = pd.DataFrame(X)
        df_labels = pd.DataFrame(y)
        
        if not os.path.exists(self.path_to_data_dir) :
            os.mkdir(self.path_to_data_dir)
            
        df_features.to_csv(os.path.join(self.path_to_data_dir, "features.csv"), header=False, index=False)
        df_labels.to_csv(os.path.join(self.path_to_data_dir, "labels.csv"), header=False, index=False)
=======
        self.path_to_data = os.path.join(os.getcwd(), "DATA")
        self.path_to_labels = "/home/martinho/Documents/Electric_sense_for_burried_objects_locating_classification/Own_article_script/DATA/labels.csv"
        
    
    def features_extraction(self,z=5):
        
        #------------------------features ------------------
        traj_dipole_value = self.extract_dipole_value_traji(range(17), z=z)
        X=np.empty(shape=(2400,))
        for list_dipole in traj_dipole_value :
            
            intensity_traj_i=np.empty(shape=(1,))
            for dp_i in list_dipole[0:8] : 
                intensity_traj_i = np.concatenate((intensity_traj_i, dp_i[:300,5]), axis = 0) 
            
            # intensity_traj_i_normalized = features_normalisation(intensity_traj_i[1:])
            intensity_traj_i = intensity_traj_i[1:]
            
            X = np.concatenate((X, intensity_traj_i ), axis=0)       
        
        X = X[2400:].reshape(17, 2400)
        
        df_features = pd.DataFrame(X)
         
        if not os.path.exists(self.path_to_data) :
            os.mkdir(self.path_to_data)
            
        df_features.to_csv(os.path.join(self.path_to_data, "features.csv"), header=False, index=False)
        
        #------------------------labels ------------------
        
        labels = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0]).reshape((17,1))
        
        df_labels = pd.DataFrame(labels)
        
        df_labels.to_csv(os.path.join(self.path_to_data, "labels.csv"), header=False, index=False)
>>>>>>> 2377358f6f5ed7feb032a74721058403d42f83b3
        
        self.features = df_features
        self.labels = df_labels
        
        
<<<<<<< HEAD
    def data_split(self):
        
        self.features =  pd.read_csv(self.path_to_features, header=None, index_col=None)
        self.labels = pd.read_csv(self.path_to_labels, header=None, index_col=None)
        
        print("features shape   :" , self.features.shape)
        
        X_train, X_test, y_train, y_test = train_test_split( self.features , self.labels, test_size=0.33, shuffle = True, random_state=42)
        
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64).reshape((X_train.shape[0], ))
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64).reshape((X_test.shape[0], )) 
        print("end")
        
        return X_train, y_train, X_test, y_test
      

    def find_best_learning_params(self, model, type ='lr'):
        
        X_train, y_train, _, _ = self.data_split()
        if type == 'lr' :
            C = [0.0001, 0.001, 0.01, 0.1, 1, 10]
            penalty = ['l1', 'l2', 'elasticnet', 'none']
            hyper_param_grid = dict(C = C, penalty=penalty)
        elif type == 'nn' :
            hidden_layer_sizes = [(100,), (100,100), (100,100,100)]
            activation = ['logistic', 'tanh', 'relu']
            alpha = [0.0001, 0.001, 0.01,0.1,1,10]
            hyper_param_grid = dict(hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha)
        elif type == 'rf'  :
            n_estimators   = [1, 10, 100, 500, 1000] # The number of trees in the forest.
            max_depth      = [5, 10, 20]
            min_samples_leaf  = [1, 5, 10]
            # max_leaf_nodes = [5,] % qui correspond au nombre limite de ramifications
            # min_samples_split = [2, 10, 100] qui est la profondeur maximale d'un arbre
            hyper_param_grid = dict(n_estimators = n_estimators, max_depth = max_depth,min_samples_leaf = min_samples_leaf)
            
        elif type == 'svm'  :
            C = [1e3, 5e3, 1e4, 5e4, 1e5, 1e6] 
            gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5]
            kernel = [ 'rbf', 'linear', 'poly' ]

            hyper_param_grid = {'C': C, 'gamma': gamma, 'kernel': kernel}
        
        
        elif type == 'nb'  :
            hyper_param_grid = { 'var_smoothing': np.logspace(0,-9, num=100)}
            
        elif type == 'et'  :
            hyper_param_grid = { 'n_estimators':  [int(x) for x in np.arange(start = 100, stop = 500, step = 100)],'max_depth': [2,8,16,32,50],
                           # 'min_sample_split': [2,4,6],'min_sample_leaf': [1,2],#'oob_score': [True, False],
                            'max_features': ['auto','sqrt','log2'], #'bootstrap': [True, False],'warm_start': [True, False], 'criterion': ['mse', 'mae'],
                            }
                 
        elif type == 'dt'  :      
            
            hyper_param_grid = {
                'max_depth': [2, 3, 5, 10, 20],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                'criterion': ["gini", "entropy"]
            }
            
        else :
            print("model type incorrect")
        #---------------------------------------
        
        cv = GridSearchCV(model, param_grid=hyper_param_grid, cv=7, n_jobs=-1, verbose=0)
        
        cv.fit(X_train, y_train)
        
        self.best_params = cv.best_params_
        print(self.best_params )



if __name__ == '__main__' :

    artificial_intelligence = Artificial_intelligence()
    
    # artificial_intelligence.features_extraction_segment(segment_width=10) 
    # artificial_intelligence.features_extraction(window_h=1) 
    # artificial_intelligence.data_augmentation() 
       
    
    
    X_train, y_train, X_test, y_test = artificial_intelligence.data_split()
    
    lr = False
    nn = False
    svm_ = False
    rf = False
    nb = False
    et = True
    dt = False
    
    # clf_lr = make_pipeline(preprocessing.StandardScaler(), LogisticRegression())
    # clf_lr = make_pipeline(preprocessing.MinMaxScaler(), LogisticRegression())
    
    # logistic regression
    if lr :
        clf = LogisticRegression()
        type = 'lr'
    # # Neural Network
    if nn :
        clf = MLPClassifier()
        type = 'nn'
    #SVM
    if svm_ :
        clf  = svm.SVC()
        type = 'svm'

    # # Random Forest
    if rf :
        clf  = RandomForestClassifier()
        type = 'rf'

    # # Naives bayes
    if nb :
        clf  = GaussianNB()
        type = 'nb'
    
    # # ExtraTreesClassifier
    if et :
        clf = ExtraTreesClassifier()
        type = 'et'
    
    # # Decision tree
    if dt :
        clf    = DecisionTreeClassifier()    
        type = 'dt'
    # 
    # #-------------------hyperparameters -----------------
    artificial_intelligence.find_best_learning_params(clf, type=type)
    clf.set_params(**artificial_intelligence.best_params)
    # #------------------- 
    
    clf.fit(X_train, y_train) 
    
    print(" prediction score : ", clf.score(X_test, y_test))
    
    
    # print("\nMatrice de confusion normalisé\n")
    
    # # fig, axes = plt.subplots(1,2)
    # plot_confusion_matrix(clf, X_test, y_test, normalize='true')
    
    
    print("Matrice de confusion non normalisé")
    plot_confusion_matrix(clf, X_test, y_test)
    
    if dt:
        fig = plt.figure()
        _ = tree.plot_tree(clf,
                        feature_names=pd.DataFrame(X_train).columns,
                        class_names=['Pipe', "No pipe"],
                        filled=True)
    
    plt.show()
    
=======

    
    
def features_normalisation(features):
    
    min = np.min(features, axis=0)
    max = np.max(features, axis=0)
    
    features_normalized = (features - min)/(max - min)

    return features_normalized

if __name__ == '__main__' :

    arti_intelli = Artificial_intelligence()
    arti_intelli.features_extraction() 
    X_train, X_test, y_train, y_test = train_test_split( arti_intelli.features , arti_intelli.labels, test_size=0.33, shuffle = True, random_state=42)
    
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64).reshape((X_train.shape[0], ))
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64).reshape((X_test.shape[0], )) 
    
    # clf_lr = make_pipeline(preprocessing.StandardScaler(), LogisticRegression())
    clf_lr = make_pipeline(preprocessing.MinMaxScaler(), LogisticRegression())
    clf_lr.fit(X_train, y_train) 
    clf_lr.score(X_test, y_test) 
    
    print(" prediction score : ", clf_lr.score(X_test, y_test))
    
        
    
    print("Matrice de confusion normalisé")
    plot_confusion_matrix(clf_lr, X_test, y_test, normalize='true')

    print("Matrice de confusion non normalisé")
    plot_confusion_matrix(clf_lr, X_test, y_test)

    
    # # Neural Network
    # clf_nn = MLPClassifier()
    # clf_nn.fit(X_train, y_train)
    # print(" prediction score : ", clf_nn.score(X_test, y_test)) 
    
    # # SVM
    # clf_svm = svm.SVC()
    # clf_svm.fit(X_train, y_train)
    # print(" prediction score : ", clf_svm.score(X_test, y_test))


    # # Random Forest
    # clf_rf = RandomForestClassifier()
    # clf_rf.fit(X_train, y_train)
    # print(" prediction score : ", clf_rf.score(X_test, y_test))
    
    
    #     #--------------------- prediction 
    Y_pred_lr = clf_lr.predict(X_test)
    
    print(Y_pred_lr, y_test)
    # Y_pred_nn = clf_nn.predict(X_test)
    # Y_pred_svm = clf_svm.predict(X_test)
    # Y_pred_rf= clf_rf.predict(X_test)
    plt.show()
>>>>>>> 2377358f6f5ed7feb032a74721058403d42f83b3
