# Estimation DoB

# ------Importation

# %%
from cmath import pi
import matplotlib.pyplot as plt
import os
import scipy.ndimage.filters as filters
from time import time
import numpy as np
import scipy as sp
import pandas as pd
from scipy import interpolate

import utils.augmentation as aug
import utils.helper as hlp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
    
pd.options.display.max_colwidth = 600


class Data_extraction:

    def __init__(self, ESSAI=2, TEST=2, PIPE = None) -> None:

        if (ESSAI == 1):

            if TEST == 2:
                self.path = os.path.abspath(
                    "../Datasets/ESSAIS 1/3BO-COND/Eau_salee/TEST 2")
                self.path_altitude_echosondeur = os.path.join(
                    self.path, "Pipe 1 - BO = 1 cm/Echosondeur/Test2-Pipe1(BO=1cm)")
                self.TEST = 2
                self.traj = 17
                self.pipe = PIPE

            elif TEST == 1:
                self.path = os.path.abspath(
                    "../Datasets/ESSAIS 1/3BO-COND/Eau_salee/TEST 1")
                self.TEST = 1
                self.traj = 13
                self.pipe = PIPE

            elif TEST == 3:
                self.path = os.path.abspath(
                    "../Datasets/ESSAIS 1/3BO-COND/Eau_salee/TEST 3")
                self.path_altitude_echosondeur = os.path.join(
                    self.path, "Pipe 1 - BO = 1 cm/Echosondeur/Test2-Pipe1(BO=1cm)")
                self.TEST = 3
                self.traj = 3
                self.pipe = PIPE
                
            self.v = 4
            self.path_altitude_echosondeur = os.path.join(
                self.path, "Pipe 1 - BO = 1 cm/Echosondeur/Test2-Pipe1(BO=1cm)")

        elif ESSAI == 2:

            if TEST == 1:
                self.path = os.path.abspath(
                    "../Datasets/ESSAIS 2/Tests Sol Ratissé - 3 BO/TEST 1")
                self.TEST = 1
                self.traj = 13
                self.pipe = PIPE

            elif TEST == 2:
                self.path = os.path.abspath(
                    "../Datasets/ESSAIS 2/Tests Sol Ratissé - 3 BO/TEST 2")
                self.TEST = 2
                self.traj = 17
                self.pipe = PIPE
                
                pipe_path = ("TOF Test2 - Alt = 5cm\Pipe 1 - BO = 1 cm", 
                             "TOF Test2 - Alt = 5cm\Pipe 2 - BO = 10 cm",
                             "TOF Test2 - Alt = 5cm\Pipe 3 - BO = 20 cm")[self.pipe - 1]
                
                
                self.path_altitude_echosondeur = os.path.join(os.path.split(self.path)[0], 'TOF', pipe_path )
   
            elif TEST == 3:
                self.path = os.path.abspath(
                    "../Datasets/ESSAIS 2/Tests Sol Ratissé - 3 BO/TEST 3")
                self.TEST = 3
                self.traj = 3
                self.pipe = PIPE
                
            self.v = 4

        print(" ESSAI {} and TEST {} ".format(ESSAI, TEST))

        self.path_to_extracted = os.path.join(os.getcwd(), "EXTRACTED")
        self.path_to_data_dir = os.path.join(os.getcwd(), "DATA")

        self.save_altitude_data = False

        # echantillonage
        self.ech_cut0 = [129, 166, 143, 132, 161, 153, 177,
                         128, 120, 121, 122, 134, 132, 130, 124, 122, 122]
        self.ech_cutf = [477, 515, 496, 480, 507, 505, 530,
                         477, 468, 482, 471, 495, 481, 479, 474, 488, 471]
        self.dipole = [12, 13, 16, 18, 26, 27, 28, 36, 37, 38, 45, 58, 68]

        self.traj_case = {"pipe" : range(5,10),
                          "no_pipe" : list(range(2))+list(range(15,17)),
                          "all_traj" : range(17), 
                            }
        
        self.index_traj = list(self.traj_case.keys())[2]
        
        
    # [data1, data2, data3 ... data_tejnum]

    def extract_data_frome_file(self, z, verbose=False):

        all_data = []
        path_z = []
        data_z = []

        if self.TEST == 2:

            path_pipes = sorted(os.listdir(self.path), key=hlp.key_pipe)
            choice = [" ", "Les dossiers dans {} sont {}".format(
                self.path.split('/')[-1],  path_pipes)][verbose]
            print(choice)

            path_pipe = os.path.join(self.path, path_pipes[self.pipe-1])

            alt_dir = sorted(os.listdir(path_pipe), key=hlp.key_alt)

            choice = [" ", "Les dossiers dans {} sont {}".format(
                path_pipe.split('/')[-1], alt_dir)][verbose]
            print(choice)

            self.z = [hlp.key_alt(rep_alt) for rep_alt in alt_dir]
            for i in range(len(alt_dir)):
                path_z.append(os.path.join(path_pipe, alt_dir[i]))

            for i in range(len(alt_dir)):
                data_z.append(sorted(os.listdir(path_z[i]), key=hlp.key_data))

        elif self.TEST == 1:

            # retirer le dossier old, tous les alts
            alt_dir = sorted(os.listdir(self.path))[:-1]
            alt_dir = sorted(alt_dir, key=hlp.key_alt)

            self.z = [hlp.key_alt(rep_alt) for rep_alt in alt_dir]

            for i in range(len(alt_dir)):

                path_data = sorted(os.listdir(
                    os.path.join(self.path, alt_dir[i])))
                if len(path_data) >= 14:
                    path_data = path_data[1:]

                path_z.append(os.path.join(self.path, alt_dir[i]))
                data_z.append(sorted(path_data, key=hlp.key_data))
        
        if self.TEST == 3:

            path_pipes = sorted(os.listdir(self.path), key=hlp.key_pipe)
            choice = [" ", "Les dossiers dans {} sont {}".format(
                self.path.split('/')[-1],  path_pipes)][verbose]
            print(choice)

            path_pipe = os.path.join(self.path, path_pipes[self.pipe-1])

            data_z = sorted(os.listdir(path_pipe), key=hlp.key_data)

            choice = [" ", "Les dossiers dans {} sont {}".format(
                path_pipe.split('/')[-1], data_z)][verbose]
            print(choice)
            self.z = [None]
            
            
        choice = [" ", "les differentes altitudes sont : {}".format(
            self.z)][verbose]
        print(choice)
        
        # extraction des donnees
        
        if self.TEST == 3 :
            all_traj =  data_z
            path = path_pipe
        else :
            all_traj =  data_z[self.z.index(z)]
            path = path_z[self.z.index(z)]
        
        for traj in all_traj:
            
            data = pd.read_csv(os.path.join(
                path, traj), on_bad_lines='skip', dtype='unicode')
            
            data.head()
            data = data.iloc[2:, :6]
            data.drop_duplicates(inplace=True)
            data = np.array(data, dtype=np.float64)
            all_data.append(data)

        return all_data

    # [ [ dp1, dp2 ...dp13], ...........[ dp1, dp2 ...dp13], ]
    def extract_dipole_value_traji(self,  num_traj_list, z):

        def echantillonnage(values, min):

            values_ech = []
            for data_traj in values:
                list_dp = []
                for data_dp in data_traj:
                    list_dp.append(np.concatenate(
                        (data_dp[1:min, 0].reshape(-1, 1), data_dp[1:min, 5].reshape(-1, 1)), axis=1))  # or df.drop
                values_ech.append(list_dp)
            return values_ech

        def cutting(values, x_val=np.linspace(40, 160, 300)):

            values_cut = []
            for data_traj in values:
                list_dp = []
                for data_dp in data_traj:
                    tsec = data_dp[:, 0]*1e-3

                    if self.TEST == 2:
                        dist_y = (tsec - tsec[0])*self.v
                        y_val = np.interp(x_val, dist_y, data_dp[:, 5])
             
             
                    elif self.TEST == 1:
                        dist_y = (tsec - tsec[0])*self.v
                        
                        y_val = np.interp(x_val, dist_y, data_dp[:, 5])
                        
                    elif self.TEST == 3:
                        dist_y = (tsec - tsec[0])*self.v
                        y_val = np.interp(x_val, dist_y, data_dp[:, 5])
                        
                    dp = np.concatenate(
                        (x_val.reshape(-1, 1), y_val.reshape(-1, 1)), axis=1)
                    list_dp.append(dp)

                values_cut.append(list_dp)

            return values_cut

        traj_dipole_value = []
        all_data = self.extract_data_frome_file(z, verbose=True)

        signal_shape = []
        for num_traj in num_traj_list:
            data = all_data[num_traj]

            # A = np.empty((0,10))A = np.empty((0,10))
            dp12 = data[1, :].reshape(1, 6)
            dp13 = data[1, :].reshape(1, 6)
            dp16 = data[1, :].reshape(1, 6)
            dp18 = data[1, :].reshape(1, 6)
            dp26 = data[1, :].reshape(1, 6)
            dp27 = data[1, :].reshape(1, 6)
            dp28 = data[1, :].reshape(1, 6)
            dp36 = data[1, :].reshape(1, 6)
            dp37 = data[1, :].reshape(1, 6)
            dp38 = data[1, :].reshape(1, 6)
            dp45 = data[1, :].reshape(1, 6)
            dp58 = data[1, :].reshape(1, 6)
            dp68 = data[1, :].reshape(1, 6)

            for i in range(0, data.shape[0]): # a refaire plus propre avec pandas 
                if data[i, 1] == 1.0 and data[i, 2] == 2.0:
                    dp12 = np.concatenate(
                        (dp12, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 1.0 and data[i, 2] == 3.0:
                    dp13 = np.concatenate(
                        (dp13, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 1.0 and data[i, 2] == 6.0:
                    dp16 = np.concatenate(
                        (dp16, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 1.0 and data[i, 2] == 8.0:
                    dp18 = np.concatenate(
                        (dp18, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 2.0 and data[i, 2] == 6.0:
                    dp26 = np.concatenate(
                        (dp26, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 2.0 and data[i, 2] == 7.0:
                    dp27 = np.concatenate(
                        (dp27, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 2.0 and data[i, 2] == 8.0:
                    dp28 = np.concatenate(
                        (dp28, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 3.0 and data[i, 2] == 6.0:
                    dp36 = np.concatenate(
                        (dp36, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 3.0 and data[i, 2] == 7.0:
                    dp37 = np.concatenate(
                        (dp37, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 3.0 and data[i, 2] == 8.0:
                    dp38 = np.concatenate(
                        (dp38, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 4.0 and data[i, 2] == 5.0:
                    dp45 = np.concatenate(
                        (dp45, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 5.0 and data[i, 2] == 8.0:
                    dp58 = np.concatenate(
                        (dp58, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 6.0 and data[i, 2] == 8.0:
                    dp68 = np.concatenate(
                        (dp68, data[i, :].reshape(1, 6)), axis=0)

            signal_shape.extend([dp12.shape[0], dp13.shape[0], dp16.shape[0], dp18.shape[0], dp26.shape[0],
                                dp27.shape[0], dp28.shape[0], dp36.shape[0], dp37.shape[0], dp38.shape[0],
                                dp45.shape[0], dp58.shape[0], dp68.shape[0]])

        
            # traj_dipole_value.append(
            #     [dp13, dp16, dp36])
            
            # traj_dipole_value.append(
            #     [dp13, dp16, dp36, dp18, dp38, dp68])
            
          
            # traj_dipole_value.append(
            #     [dp13, dp16, dp36, dp18, dp38, dp68, dp26, dp28]) 
            
            
            # traj_dipole_value.append(
            #      [dp13, dp16, dp36, dp18, dp38, dp68, dp26, dp28, dp12]) 
            
            
            # traj_dipole_value.append(
            #      [dp13, dp16, dp36, dp18, dp38, dp68, dp26, dp28, dp12, dp27]) 
            
            
            # traj_dipole_value.append(
            #     [dp13, dp16, dp36, dp18, dp38, dp68, dp26, dp28 , dp12 , dp27, dp37, dp45, dp58])
            
            # traj_dipole_value.append(
            #     [dp12])
            
            traj_dipole_value.append(
                [dp12, dp13, dp16, dp18, dp26, dp27, dp28, dp36, dp37, dp38, dp45, dp58, dp68])
 
 
            # traj_dipole_value.append(
            #      [dp13]) 
        # --------echantillonnage
        min_signal_shape = min(signal_shape)

        print(" min ", min_signal_shape)
        # print(" min signal shape = " , min_signal_shape)
        # traj_dipole_value = echantillonnage(traj_dipole_value, min_signal_shape)

        # --------interpolation
        x_val = [np.linspace(40, 460, 1064), np.linspace(
            40, 150, 300), np.linspace(5, 50, 432)][self.TEST - 1]

        traj_dipole_value = cutting(traj_dipole_value, x_val)
        
        # ---------------------

        # plt.plot((dp13[:,0]-dp13[0,0])*self.v*1e-3, dp13[:,5])
        
        # plt.show()
        
        return traj_dipole_value

    def save_data_z(self, z=5):

        traj_dipole_value = self.extract_dipole_value_traji(
            self.traj_case[self.index_traj], z=z)
                   
        num_traj = len(traj_dipole_value)

        num_dipole = len(traj_dipole_value[0])
        signal_shape = traj_dipole_value[0][0].shape[0]-1

        print("signal shape  : ", signal_shape)
        print("Nombre de trajectoire   : ", num_traj)
        print("Nombre de dipole   : ", num_dipole)

        X = np.zeros((num_traj, num_dipole*signal_shape))

        for i, data_traj in enumerate(traj_dipole_value):
            dp = np.array([])
            for data_dp in data_traj:
                _, integrale_derivee_I = hlp.integrale_derivee(
                    data_dp[:, 0], data_dp[:, 1])
                # integrale_derivee_I = data_dp[1:, 1]   
                
                dp = np.concatenate((dp, integrale_derivee_I))
            X[i] = dp.ravel()

        # y  =  np.array([0]*5 + [1]*8 + [0]*4).reshape((num_traj,1))
        if self.TEST == 1:
            
            temp_traj = [0]*4 + [1]*4 + [0]*5
            y = []
            for cpt in self.traj_case[self.index_traj] :
                y.append(temp_traj[cpt])
            y = np.array(y).reshape((num_traj, 1))
            
        elif self.TEST == 2:
            
            temp_traj = [0]*5 + [1]*8 + [0]*4
            
            y = []
            for cpt in self.traj_case[self.index_traj] :
                y.append(temp_traj[cpt])
            y = np.array(y).reshape((num_traj, 1))

        # ajout de la valeur d'echantiollonnage
        X = np.concatenate(
            (X, np.array([signal_shape]*num_traj).reshape(-1, 1)), axis=1)
        X = np.concatenate((X, y), axis=1)  # ajout des labels

        df = pd.DataFrame(X)
        if not os.path.exists(self.path_to_data_dir):
            os.mkdir(self.path_to_data_dir)

        df.to_csv(os.path.join(self.path_to_data_dir,
                  "data{}.csv".format(z)), header=False, index=False)

    def interpolation_courbe(self, list_signal,  alt_z, alt_z_val):

        signal_allz = []
        for i in range(list_signal[0].shape[0]):
            I_z = []
            for signal in list_signal:
                I_z.append(signal[i])
            # I_allz = np.interp(alt_z_val, alt_z, I_z)
            # signal_allz.append(I_allz)

            I_allz = hlp.cubicinterpolation(alt_z_val, alt_z, I_z)
            signal_allz.append(I_allz)
            
            
        # signal pour toutes les altitudes
        return np.array(signal_allz, dtype=np.float64).T
        
    def extract_dipole_value_all_altitude(self, alt_z, alt_z_val):

        data = []
        X_no_interp = []

        for z in alt_z:
            data.append(pd.read_csv(os.path.join(self.path_to_data_dir,
                        'data{}.csv'.format(z)), header=None, index_col=None))

        signal_shape_array = data[0].iloc[:, -2]  # array  of signal_shape
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]
        # nombre de dipole
        self.dp_n = int(data[0].shape[1] / self.signal_shape)
        self.traj = int(data[0].shape[0])

        for i in range(len(alt_z)):
            X_no_interp.append(np.array(
                data[i].iloc[:, :-2], dtype=np.float64).reshape(-1, self.dp_n, self.signal_shape))


        self.num_alt = len(alt_z_val)
        X_interp = np.zeros(
            (self.num_alt, self.traj, self.dp_n, self.signal_shape))
        y = alt_z_val.reshape(-1, 1)  # array  of signal_shape

        # #     #------------------------features + window------------------
        print(" le nombre de trajectoires est {} ".format(self.traj))
        print(" le nombre de dipoles est {} ".format(self.dp_n))
        print(" nombre d'altitude obtenue par regression {}".format(self.num_alt))

        for k in range(len(alt_z_val)):
            for i in range(self.traj):
                for j in range(self.dp_n):
                    list_signal = []
                    for x in X_no_interp:
                        list_signal.append(x[i][j])
                    X_interp[k][i][j] = self.interpolation_courbe(
                        list_signal, alt_z,  alt_z_val)[k]

        X_interp = X_interp.reshape(-1, self.traj*self.dp_n*self.signal_shape)
        X_interp = np.concatenate(
            (X_interp, np.array([self.signal_shape]*self.num_alt).reshape(-1, 1)), axis=1)
        X_interp = np.concatenate((X_interp, y), axis=1)

        df = pd.DataFrame(X_interp)

        if not os.path.exists(self.path_to_data_dir):
            os.mkdir(self.path_to_data_dir)

        df.to_csv(os.path.join(self.path_to_data_dir,
                  "data_all_z_p_{}_dp{}_{}.csv".format(self.pipe, self.dp_n, self.index_traj)), header=False, index=False)

        # fig, axes = plt.subplots(3,2, figsize=(15,12))
        # axes = axes.flatten()

        # for j in range(6):
        #     axes[j].plot(X_interp[0,j*self.signal_shape:(j+1)*self.signal_shape])

        # plt.show()
        print("file for all altitude created")

    def plot_dipole_traji_dipolej(self, num_traj_list, num_dipole_list, z=5, axis_x=True):

        traj_dipole_value = self.extract_dipole_value_traji(num_traj_list, z=z)
    
        for traj, list_dipole in enumerate(traj_dipole_value):
            fig = plt.figure()
            sns.set()

            for i in num_dipole_list:

                dp = list_dipole[i]

                x = dp[:, 0]
                tsec = range(dp.shape[0])

                x = (tsec, x)[axis_x]

                # integrale_derivee du signal
                X, Y = hlp.integrale_derivee(x, dp[:, 1])
                # X, Y = dp[1:, 0], dp[1:, 1]
                

                # plt.plot(X, Y,  label="dp{}".format(self.dipole[i]), linewidth=1)

                # Moyenne glissante
                lissage_dp = hlp.lissage(list(Y), 20)

                # plt.plot(X, Y, label="dp sans filtre{}".format(
                # self.dipole[i]), linewidth=1)   

                plt.plot(X, lissage_dp, label="lissage_dp{}".format(
                self.dipole[i]), linewidth=1)              

            choice = ([" times "] + ([[" X (cm)"], ["Y(cm) "], ["Z(cm) "]]
                      [self.TEST - 1]))[axis_x]
            plt.xlabel(choice)
            plt.ylabel("I_rms")
            plt.title(" Traj : {}, alt : {} cm, pipe : {} ".format(
                num_traj_list[traj]+1, z, self.pipe))
            # plt.grid()
            plt.legend()

        # plt.show()

    

    def plot_cartographie_v2(self, num_dipole_list, z=5,  axis_x=True):
        
        x0 = 75 # position x initial du test
        tn = [0,5,10,15,17,19,21,23,25,27,29,31,33,35,40,45,50]
        px = np.array(tn) + x0
        y_integ_int = np.linspace(40,160,300) 
        
        current_all = []
        traj_dipole_value = self.extract_dipole_value_traji(range(self.traj), z=z)
        
        for i in num_dipole_list:
            Idint = []
            for traj, list_dipole in enumerate(traj_dipole_value):
                dp = list_dipole[i]
                Idint.append(dp[:,1])
            current_all.append(np.array(Idint))
                
        [X,Y] = np.meshgrid(y_integ_int,px) 

        for dp, Idint in enumerate(current_all) :
            plt.figure(dpi=200)
            plt.title("cartographie : T = {},  dipoles : {}, Alt : {}cm".format(self.TEST,self.dipole[dp], z))
            plt.contourf(X,Y,Idint,cmap='jet',levels = 200)
            plt.xticks(ticks=np.arange(60, 170, 30))
            plt.xlabel('Y (cm)')
            plt.ylabel('X (cm)')
            fmt_sci = '%0.5f'
            plt.gca().set_aspect("equal")
            plt.colorbar()
            plt.tight_layout()
            # plt.gca().invert_xaxis()
            plt.show()

            plt.show()
            
    def plot_cartographie_v1(self, num_dipole_list, z=5,  axis_x=True):
        traj_dipole_list = self.extract_dipole_value_traji(
            range(self.traj), z=z)  # traj_show = 17
        for i in num_dipole_list:
            fig = plt.figure()

            for j in range(len(traj_dipole_list)):
                traj_1 = traj_dipole_list[j]
                dp = traj_1[i]

                irms_max = np.max(dp, axis=0)[1]
                irms_min = np.min(dp, axis=0)[1]

                x = dp[:, 0]
                tsec = range(dp.shape[0])
                x = (tsec, x)[axis_x]

                X, Y = hlp.integrale_derivee(
                    x, (dp[:, 1] - irms_min) / (irms_max-irms_min))

                plt.scatter(
                    x[:-1], [j for i in np.arange(0, 3-0.01, 0.01)], c=Y, cmap='viridis')

            (plt.xlabel("time(msss)"), plt.xlabel("Distance(cm)"))[axis_x]
            plt.ylabel("trajectoires")
            plt.title(" Dipole{}".format(self.dipole[i]))

            plt.legend()
            plt.colorbar(orientation="horizontal",
                         fraction=0.07, anchor=(1.0, 0.0))
        plt.show()
        
    def plot_altitude_echosondeur(self, list_num_traj, ech=True, axis_x=True) :

        print(self.path_altitude_echosondeur)
        list_traj_altitude = sorted(os.listdir(self.path_altitude_echosondeur))
        

        for e, i in enumerate(list_num_traj) :
            # fig = plt.figure()
            data_i = pd.read_csv(os.path.join(self.path_altitude_echosondeur, list_traj_altitude[i]), header=None, index_col=None, dtype="float64")
            data_i = np.array(data_i.iloc[2:,:], dtype=np.float64)


            tsec = [data_i[:, :4],data_i[self.ech_cut0[e]:self.ech_cutf[e], :4]][ech]@np.array([[3600],[60],[1],[1e-3]]) # equivalent a if ech=True : on ecahntillonne
            altitude = (data_i[:, 4],data_i[self.ech_cut0[e]:self.ech_cutf[e], 4])[ech]                                    # elsee : on echantillonne pas

            dist_y = (tsec - tsec[0])*self.v

            x = (tsec, dist_y+14.5)[axis_x]

            #-------------------- sauvegarde des donnees ---------------
            if self.save_altitude_data :
                if not os.path.exists(self.path_to_extracted):
                    os.mkdir(self.path_to_extracted)

                df_altitude = pd.DataFrame(np.concatenate((tsec, data_i[:,4].reshape(-1, 1)), axis=1))
                df_altitude.to_csv(os.path.join(self.path_to_extracted,
                        "trajec_altitude_{}_{}.csv".format(i, int(ech))), header=False, index=False)

            #----------------------------------------------------------
            # plt.plot(tsec, altitude, label="traj{}".format(i), linewidth=1)
            plt.plot(x, altitude, ms = 1,label="traj{}".format(i), linewidth=1)
            (plt.xlabel("time(s)"), plt.xlabel(" Y(cm) "))[axis_x]
            plt.ylabel(" Altitude(cm) ")
            plt.legend()

    def generate_data_for_interp(self):
       
        essai = 2
        
        if essai == 1 :
            for z in range(5,25,5):
                self.save_data_z(z=z)
            pas = 0.5
            alt_z_val = np.arange(5, 20+pas, pas)  # TEST 2 ESSAI 1
            alt = range(5,25,5) 
    
            self.extract_dipole_value_all_altitude(alt_z=alt, alt_z_val=alt_z_val)

            
            
        elif essai == 2 :


            # alt = range(4,16,4)
            # for z in alt:
            #     self.save_data_z(z=z)

            # pas = 2
            # alt_z_val = np.arange(4, 12+pas, pas) # TEST 2 ESSAI 2
    
            # self.extract_dipole_value_all_altitude(alt_z=alt, alt_z_val=alt_z_val)
        
            alt = range(4,14,2)
            
            for z in alt:
                self.save_data_z(z=z)

            pas = 2
            alt_z_val = np.arange(4, 12+pas, pas) # TEST 2 ESSAI 2
    
            self.extract_dipole_value_all_altitude(alt_z=alt, alt_z_val=alt_z_val)
            
            
           
    def extract_alt(self, gaussian_filter=True):
                
        all_data = []
        slice_traj = []
        all_traj = sorted(os.listdir(self.path_altitude_echosondeur), key=hlp.key_data)
        
        
        for cpt in self.traj_case[self.index_traj] :
            slice_traj.append(all_traj[cpt])
            
        for traj in slice_traj:
            
            data = pd.read_csv(os.path.join(
                self.path_altitude_echosondeur, traj), on_bad_lines='skip', dtype='unicode')
            
            data.head()
            data.drop_duplicates(inplace=True)
            data = np.array(data, dtype=np.float64)
            
            x_val, y_val  = interpolation_alt(np.linspace(40, 150, 300), data)
            
            if gaussian_filter :
                y_val = filters.gaussian_filter1d(y_val,sigma=5)
            all_data.append(y_val)
            
            
        df =  pd.DataFrame(np.array(all_data))

        df.to_csv(os.path.join(self.path_to_data_dir,
         "alt_p_{}_{}.csv".format(self.pipe,self.index_traj)), header=False, index=False)
        
        
        print(f"nombre de trajectoires selectionnées pour l'altitude = {len(self.traj_case[self.index_traj])}")
        return all_data

def interpolation_alt(x_val, alt) : 
        
    dist_y = (alt[:,0] - alt[0,0])*4e-3 
    y_val = np.interp(x_val, dist_y, alt[:, 2])*1e-1
        
    return x_val, y_val





# %% define a box
if __name__ == '__main__':

    data_extraction = Data_extraction(ESSAI = 2, TEST=1, PIPE=1)
    
    # alt = data_extraction.extract_alt()
    
    # x = np.linspace(40, 150, 300)
    
    # data_extraction.plot_dipole_traji_dipolej(range(5,8), [1,10,12], z = 4)
    # data_extraction.plot_dipole_traji_dipolej(range(5,8), [2,5,9], z = 4)
    # plt.show()
    
    # data_extraction.plot_cartographie_v2(range(13), z = 5)
    
    # interpolation
    # data_extraction.generate_data_for_interp()
    # data_extraction.save_data_z(z=4)
# %%
