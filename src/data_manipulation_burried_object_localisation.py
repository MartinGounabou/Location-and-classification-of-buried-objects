# Estimation DoB

# ------Importation

from email import header
from sqlite3 import Row
from cv2 import drawFrameAxes
import matplotlib.pyplot as plt
import os
from time import time
import numpy as np
import scipy as sp
import pandas as pd
from scipy import interpolate

import utils.augmentation as aug
import utils.helper as hlp


pd.options.display.max_colwidth = 600


class Data_extraction:

    def __init__(self) -> None:
        
        self.path = os.path.abspath("../Datasets/ESSAIS/3BO-COND/Eau_salee/TEST 2")

        self.path_altitude_echosondeur = os.path.join(self.path,"Pipe 1 - BO = 1 cm/Echosondeur/Test2-Pipe1(BO=1cm)")

        self.path_to_extracted = os.path.join(os.getcwd(), "EXTRACTED")
        
        self.save_altitude_data = False

        #echantillonage
        self.ech_cut0 = [129,166,143,132,161,153,177,128,120,121,122,134,132,130,124,122,122]
        self.ech_cutf = [477,515,496,480,507,505,530,477,468,482,471,495,481,479,474,488,471]
        self.dipole = [12,13,16,18,26,27,28,36,37,38,45,58,68]
        
        self.z = [5,10,15,20]

        self.path_to_data_dir = os.path.join(os.getcwd(), "DATA")
        
        self.vx = 4        
        
    def extract_data_frome_file(self, z, pipe):

        all_data = []
        path_pipes = sorted(os.listdir(self.path), key=hlp.key_pipe)

        path_pipe1 = os.path.join(self.path, path_pipes[0])
        path_pipe2 = os.path.join(self.path, path_pipes[1])
        path_pipe3 = os.path.join(self.path, path_pipes[2])

        path_pipe = [path_pipe1, path_pipe2,path_pipe3][pipe-1]
        
        

        
        alt_pipe_dir = sorted(os.listdir(path_pipe), key=len)[:-1] # trie suivant la taille et retire le dossier echosondeur
        alt_pipe_dir = sorted(alt_pipe_dir, key=hlp.key_alt)

        path_pipe_5cm = os.path.join(path_pipe, alt_pipe_dir[0])
        path_pipe_10cm = os.path.join(path_pipe, alt_pipe_dir[1])
        path_pipe_15cm = os.path.join(path_pipe, alt_pipe_dir[2])
        path_pipe_20cm = os.path.join(path_pipe, alt_pipe_dir[3])
        path_pipe_z = [path_pipe_5cm, path_pipe_10cm,path_pipe_15cm,path_pipe_20cm]
        data_pipe_5cm = sorted(os.listdir(path_pipe_5cm), key=hlp.key_data)
        data_pipe_10cm = sorted(os.listdir(path_pipe_10cm), key=hlp.key_data)
        data_pipe_15cm = sorted(os.listdir(path_pipe_15cm), key=hlp.key_data)
        data_pipe_20cm = sorted(os.listdir(path_pipe_20cm), key=hlp.key_data)
        
    

        data_pipe1_z = [data_pipe_5cm, data_pipe_10cm, data_pipe_15cm, data_pipe_20cm]
        
        for traj in data_pipe1_z[self.z.index(z)]:
            data = pd.read_csv(os.path.join(path_pipe_z[self.z.index(z)], traj), header=None)
            data.head()

            data = data.iloc[3:, :6]
            data.drop_duplicates(inplace=True)

            data = np.array(data, dtype=np.float64)

            all_data.append(data)
        return all_data
    
    def extract_dipole_value_traji(self,  num_traj_list, z, pipe): #liste des trajectoires dont on veut les valeurs

        def echantillonnage(values, min) :
            
            values_ech = []
            for data_traj in values :
                list_dp = []
                for data_dp in data_traj :
                    list_dp.append(data_dp[1:min,:])
                values_ech.append(list_dp)
            return values_ech
        
        def cutting(values, x_val=np.linspace(40, 160,300)):
            values_cut = []
            for data_traj in values :
                list_dp = []
                for data_dp in data_traj :
                    tsec = data_dp[:, 0]*1e-3
                    dist_y = (tsec - tsec[0])*self.vx
                    
                    y_val = np.interp(x_val,dist_y, data_dp[:,5])
                
                    dp = np.concatenate((x_val.reshape(-1,1),y_val.reshape(-1,1)), axis =1)
                    list_dp.append(dp)
                    
                values_cut.append(list_dp)
            return values_cut
                
        traj_dipole_value = []
        all_data = self.extract_data_frome_file(z, pipe)
        
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

            for i in range(0,data.shape[0]):
                if data[i, 1] == 1.0 and data[i, 2] == 2.0:
                    dp12 = np.concatenate((dp12, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 1.0 and data[i, 2] == 3.0:
                    dp13 = np.concatenate((dp13, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 1.0 and data[i, 2] == 6.0:
                    dp16 = np.concatenate((dp16, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 1.0 and data[i, 2] == 8.0:
                    dp18 = np.concatenate((dp18, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 2.0 and data[i, 2] == 6.0:
                    dp26 = np.concatenate((dp26, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 2.0 and data[i, 2] == 7.0:
                    dp27 = np.concatenate((dp27, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 2.0 and data[i, 2] == 8.0:
                    dp28 = np.concatenate((dp28, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 3.0 and data[i, 2] == 6.0:
                    dp36 = np.concatenate((dp36, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 3.0 and data[i, 2] == 7.0:
                    dp37 = np.concatenate((dp37, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 3.0 and data[i, 2] == 8.0:
                    dp38 = np.concatenate((dp38, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 4.0 and data[i, 2] == 5.0:
                    dp45 = np.concatenate((dp45, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 5.0 and data[i, 2] == 8.0:
                    dp58 = np.concatenate((dp58, data[i, :].reshape(1, 6)), axis=0)
                if data[i, 1] == 6.0 and data[i, 2] == 8.0:
                    dp68 = np.concatenate((dp68, data[i, :].reshape(1, 6)), axis=0)
                    
            signal_shape.extend([dp12.shape[0], dp13.shape[0], dp16.shape[0], dp18.shape[0], dp26.shape[0], 
                                dp27.shape[0], dp28.shape[0], dp36.shape[0], dp37.shape[0], dp38.shape[0], 
                                dp45.shape[0], dp58.shape[0], dp68.shape[0]])
        
            traj_dipole_value.append([dp12, dp13, dp16, dp18, dp26, dp27, dp28, dp36, dp37, dp38, dp45, dp58, dp68])
        
        min_signal_shape = min(signal_shape)
        
        # traj_dipole_value = echantillonnage(traj_dipole_value, min_signal_shape)        
        
        x_val = np.linspace(40,160,300)
        traj_dipole_value = cutting(traj_dipole_value, x_val)
        
        return traj_dipole_value
    
    def save_data(self, pipe=1):
        
        traj_dipole_value = self.extract_dipole_value_traji(range(17), z=5, pipe=pipe)
        
        num_traj = len(traj_dipole_value)
        num_dipole = 13
        signal_shape = traj_dipole_value[0][0].shape[0]-1
        
        print("signal shape  : ", signal_shape)
        
        X = np.zeros((num_traj,num_dipole*signal_shape ))
        
        for i, data_traj in enumerate(traj_dipole_value) :
            dp = np.array([])
            for data_dp in data_traj :
                _,integrale_derivee_I = hlp.integrale_derivee(data_dp[:,0],data_dp[:,1])               
                dp = np.concatenate((dp, integrale_derivee_I))
            X[i] = dp.ravel()
        
        y = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0]).reshape((17,1))
        
        X = np.concatenate((X, np.array([signal_shape]*num_traj).reshape(-1,1)), axis=1) # ajout de la valeur d'echantiollonnage
        X = np.concatenate((X, y), axis=1) # ajout des labels
        
        print(X.shape)
        df = pd.DataFrame(X)
        # df = pd.DataFrame(X[0,:6*sigal_shape].reshape(6, sigal_shape).T)
        
        fig, axes = plt.subplots(3, 2, figsize=(15,12))
        axes = axes.flatten()
        
        for j in range(6):
            axes[j].plot(X[0,j*signal_shape:(j+1)*signal_shape])
            
            
        plt.show()
        
        if not os.path.exists(self.path_to_data_dir) :
            os.mkdir(self.path_to_data_dir)
            
        df.to_csv(os.path.join(self.path_to_data_dir, "data.csv"), header=False, index=False)
        
    
    def save_data_in_file(self, z=5):
        
        all_data = self.extract_data_frome_file(z)

        for i, data in enumerate(all_data):
            df = pd.DataFrame(data)
            if not os.path.exists(self.path_to_extracted):
                os.mkdir(self.path_to_extracted)
            if not os.path.exists(self.path_to_extracted+"/{}cm".format(z)):
                os.mkdir(self.path_to_extracted+"/{}cm".format(z)) 
                print("repertory '{}cm' create".format(z))               
            
            df.to_csv(os.path.join(self.path_to_extracted+"/{}cm".format(z),
                      "trajectroire_{}.csv".format(i)), header=False, index=False)
            
    def plot_dipole_traji_dipolej(self, num_traj_list, num_dipole_list, z = 5, pipe = 1, axis_x = True):

        
        traj_dipole_value = self.extract_dipole_value_traji(num_traj_list, z=z, pipe=pipe)
        
        for traj_num,list_dipole in enumerate(traj_dipole_value) :
            fig = plt.figure()
            for i in num_dipole_list:
                
                dp = list_dipole[i]
                
                x = dp[:, 0]
                tsec = range(0,300)
                
                x = (tsec, x)[axis_x]
                
                # Signal normal
                # plt.plot(x, np.tanh((dp[1:, 5]-irms_min) / ( irms_max-irms_min)),  label="dp{}".format(self.dipole[i]), linewidth=1)
                # plt.plot(x, (dp[::, 5]-irms_min) / ( irms_max-irms_min),  label="dp{}".format(self.dipole[i]), linewidth=1)
                
                # integrale_derivee du signal
                X, Y = hlp.integrale_derivee(x,dp[:, 1])
                
                irms_max = np.max(Y, axis=0)
                irms_min = np.min(Y, axis=0)
                
                plt.plot(X, Y,  label="dp{}".format(self.dipole[i]), linewidth=1)
                # Y_transform = Y.reshape((1,-1,1))
                # Y_transform = aug.scaling(Y_transform, sigma=0.4)
                # print(Y_transform.shape)
                # plt.figure()
                # plt.plot(X, Y_transform[0],  label="dp{}".format(self.dipole[i]), linewidth=1)
                
                # inversion 
                # plt.figure()
                # plt.plot(X, Y[::-1],  label="dp{}".format(self.dipole[i]), linewidth=1)
                
                # Moyenne glissante
                # plt.figure()
                # lissage_dp = hlp.lissage(list((dp[::, 5]-irms_min) / ( irms_max-irms_min)), 20)
                # plt.plot(x, lissage_dp,  label="lissage_dp{}".format(self.dipole[i]), linewidth=1)
                
            
            (plt.xlabel("time(s)"), plt.xlabel("Y(cm)"))[axis_x]
            plt.ylabel("I_rms")
            plt.title(" Traj : {}, alt : {}cm, pipe : {} ".format(num_traj_list[traj_num], z, pipe))
            plt.grid()
            plt.legend()
            # plt.savefig("pipe{}z{}.png".format(pipe, z))

        # plt.show()
    
    def plot_cartographie(self, dipole_show_list, z =5,  axis_x=True):
        traj_dipole_list = self.extract_dipole_value_traji(range(17), z=z) # traj_show = 17
        for i in dipole_show_list :
            fig = plt.figure()

            for j in range(len(traj_dipole_list)) :
                traj_1 = traj_dipole_list[j]
                dp_i =  traj_1[i]
                irms_max = np.max(dp_i, axis=0)[5]
                irms_min = np.min(dp_i, axis=0)[5]
                tsec = traj_dipole_list[0][0][:300, 0]/1000
                dist_y = (tsec -tsec[0])*4
                x = (tsec, dist_y)[axis_x]

                # plt.scatter(traj_dipole_list[0][0][:300, 0], [ j for i in np.arange(0,3,0.01)], c = np.tanh((dp_i[:300, 5]-irms_max)/irms_min), cmap='viridis')
                # plt.scatter(x, [ j for i in np.arange(0,3,0.01)], c = np.tanh((dp_i[:300, 5]-irms_min)/( irms_max-irms_min)), cmap='viridis')
                
                X, Y = hlp.integrale_derivee(x,(dp_i[:300, 5]-irms_min) / ( irms_max-irms_min) )
                
        
                plt.scatter(x[:-1], [ j for i in np.arange(0,3-0.01,0.01)], c = Y, cmap='viridis')

            (plt.xlabel("time(msss)"), plt.xlabel("Distance(cm)"))[axis_x]
            plt.ylabel("trajectoires")
            plt.title(" Dipole{}".format(self.dipole[i]))

            # plt.legend()
            plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
        plt.show()

    def plot_altitude_echosondeur(self, list_num_traj, ech=True, axis_x=True) :

        list_traj_altitude = sorted(os.listdir(self.path_altitude_echosondeur))
        traj_echosondeur = []
        for e, i in enumerate(list_num_traj) :
            # fig = plt.figure()
            data_i = pd.read_csv(os.path.join(self.path_altitude_echosondeur, list_traj_altitude[i]), header=None, index_col=None, dtype="float64")
            data_i = np.array(data_i.iloc[2:,:], dtype=np.float64)


            tsec = [data_i[:, :4],data_i[self.ech_cut0[e]:self.ech_cutf[e], :4]][ech]@np.array([[3600],[60],[1],[1e-3]]) # equivalent a if ech=True : on ecahntillonne
            altitude = (data_i[:, 4],data_i[self.ech_cut0[e]:self.ech_cutf[e], 4])[ech]                                    # elsee : on echantillonne pas

            dist_y = (tsec - tsec[0])*self.vx

            x = (tsec, dist_y+14.5)[axis_x]

            #-------------------- sauvegarde des donnees ---------------
            if self.save_altitude_data :
                if not os.path.exists(self.path_to_extracted):
                    os.mkdir(self.path_to_extracted)

                df_altitude = pd.DataFrame(np.concatenate((tsec, data_i[:,4].reshape(data_i.shape[0], 1)), axis=1))
                df_altitude.to_csv(os.path.join(self.path_to_extracted,
                        "trajec_altitude_{}_{}.csv".format(i, int(ech))), header=False, index=False)

            #----------------------------------------------------------
            # plt.plot(tsec, altitude, label="traj{}".format(i), linewidth=1)
            plot = plt.plot(x, altitude/10-7, "ro", ms = 1,label="traj{}".format(i), linewidth=1)
            (plt.xlabel("time(s)"), plt.xlabel(" Y(cm) "))[axis_x]
            plt.ylabel(" Altitude(cm) ")
            plt.legend()

        plt.show()

    def plot_dipole_traji_dipolej_echosondeur(self, num_traj_list, num_dipole_list, z=5, ech=True, axis_x=True):

        traj_echosondeur=[]
        traj_dipole_value = self.extract_dipole_value_traji(num_traj_list, z=z)
        list_traj_altitude = sorted(os.listdir(self.path_altitude_echosondeur))

        #---------------------------------------
        for e, i in enumerate(num_traj_list) :
            # fig = plt.figure()
            data_i = pd.read_csv(os.path.join(self.path_altitude_echosondeur, list_traj_altitude[i]), header=None, index_col=None, dtype="float64")
            data_i = np.array(data_i.iloc[2:,:], dtype=np.float64)


            tsec = [data_i[:, :4],data_i[self.ech_cut0[e]:self.ech_cutf[e], :4]][ech]@np.array([[3600],[60],[1],[1e-3]]) # equivalent a if ech=True : on ecahntillonne
            altitude = (data_i[:, 4],data_i[self.ech_cut0[e]:self.ech_cutf[e], 4])[ech]                                    # elsee : on echantillonne pas

            dist_y = (tsec - tsec[0])*self.vx

            x = (tsec, dist_y+14.5)[axis_x]

            traj_echosondeur.append([x, altitude/10])

        for num_traj, list_dipole in enumerate(traj_dipole_value) :
            fig = plt.figure()
            for i in num_dipole_list:
                dp = list_dipole[i]
                irms_max = np.max(dp, axis=0)[5]
                irms_min = np.min(dp, axis=0)[5]
                x = (dp[:, 0] - dp[0,0])*self.vx*1e-3
                plt.plot(x, np.tanh((dp[:, 5]-irms_min) /
                            ( irms_max-irms_min)),  label="dipole{}".format(i), linewidth=1)

                #-----------------------------
            a = traj_echosondeur[num_traj][0]
            b = traj_echosondeur[num_traj][1]
            max = np.max(b, axis = 0)
            min = np.min(b, axis = 0)
            plt.plot(a, (b-min)/(max-min), label="alt", linewidth=1)
            plt.ylabel("Alt(cm) and I")
            plt.xlabel("Y (cm) ")
            plt.title("traj{}, alt={}cm".format(num_traj, z))
            plt.legend()

        plt.show()
        #---------------------------------------------------
        

#%% define a box
if __name__ == '__main__':

    data_extraction = Data_extraction()
    
    
    
    # # data_extraction.features_labels_save(pipe=1)
    
    
    # # data_extraction.save_data_in_file(z=20) # Z = 5, 10 ,15, 20
    # # data_extraction.extract_dipole_value_traji(range(17),  z=5)
    
    # # Traitement donnees
    data_extraction.plot_dipole_traji_dipolej(range(1), range(1),  z=5, axis_x=True, pipe=1)
    plt.show()
    
    # z = 20
    # data_extraction.plot_dipole_traji_dipolej(range(4,9), range(1),  z=20)
    
    # labelisation
    # data_extraction.plot_dipole_traji_dipolej(range(17), range(8),  z=15)
    
    # else
    # data_extraction.plot_cartographie([5], z=5) # les dipoles selectionnes
    # data_extraction.plot_altitude_echosondeur(range(1))
    # data_extraction.plot_dipole_traji_dipolej_echosondeur([1,2,4],[1,4,5])
    
#%%


