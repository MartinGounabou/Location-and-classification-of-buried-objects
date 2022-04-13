# Estimation DoB

# ------Importation

from importlib.resources import path
from tkinter import E
from unittest import TestCase
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

    def __init__(self, ESSAI=1, TEST=1) -> None:
        
        if (ESSAI==1) & (TEST == 2 ):
            
            self.path = os.path.abspath("../Datasets/ESSAIS 1/3BO-COND/Eau_salee/TEST 2")
            self.path_altitude_echosondeur = os.path.join(self.path,"Pipe 1 - BO = 1 cm/Echosondeur/Test2-Pipe1(BO=1cm)")
            self.vx = 4 
            self.TEST =2
            self.traj = 17
        
            
            self.pipe = 1
            
        elif  (ESSAI ==2 ) & (TEST == 1) :
            
            self.path = os.path.abspath("../Datasets/ESSAIS 2/Tests Sol Ratissé - 3 BO/TEST 1")
            self.path_altitude_echosondeur = os.path.join(self.path,"Pipe 1 - BO = 1 cm/Echosondeur/Test2-Pipe1(BO=1cm)")
            self.vy = 3
            self.TEST=1
            self.traj = 13
            self.pipe = None
            
            
            
        print(" ESSAI {} and TEST {} ".format(ESSAI, TEST))
            
            
        self.path_to_extracted = os.path.join(os.getcwd(), "EXTRACTED")
        self.path_to_data_dir = os.path.join(os.getcwd(), "DATA")
    
        self.save_altitude_data = False

        #echantillonage
        self.ech_cut0 = [129,166,143,132,161,153,177,128,120,121,122,134,132,130,124,122,122]
        self.ech_cutf = [477,515,496,480,507,505,530,477,468,482,471,495,481,479,474,488,471]
        self.dipole = [12,13,16,18,26,27,28,36,37,38,45,58,68]
        
             
        
    def extract_data_frome_file(self, z, verbose=False): #[data1, data2, data3 ... data_tejnum]

        all_data = []
        path_z = []
        data_z = []
        
        if self.TEST==2:
            
            path_pipes = sorted(os.listdir(self.path), key=hlp.key_pipe)
            choice  = [ " " , "Les dossiers dans {} sont {}".format(self.path.split('/')[-1],  path_pipes)][verbose]
            print(choice)
            
            path_pipe = os.path.join(self.path, path_pipes[self.pipe-1])
      
            if len(os.listdir(path_pipe)) >= 5 :
                choice = [" ", "presence d'un dossier intru {}".format(sorted(os.listdir(path_pipe), key=len)[-1])][verbose]
                print(choice)
                         
                alt_dir = sorted(os.listdir(path_pipe), key=len)[:-1] # trie suivant la taille et retire le dossier echosondeur
            alt_dir = sorted(alt_dir, key=hlp.key_alt)
            
            
            choice  = [ " " , "Les dossiers dans {} sont {}".format(path_pipe.split('/')[-1],alt_dir)][verbose]
            print(choice)
            
            self.z = [hlp.key_alt(rep_alt) for rep_alt in alt_dir]
            for i in range(len(alt_dir)):
                path_z.append(os.path.join(path_pipe, alt_dir[i]))
                
            for i in range(len(alt_dir)):
                data_z.append(sorted(os.listdir(path_z[i]), key=hlp.key_data))
        
        elif self.TEST==1:

            alt_dir = sorted(os.listdir(self.path))[:-1]# retirer le dossier old, tous les alts
            alt_dir = sorted(alt_dir, key=hlp.key_alt)
            
            self.z = [hlp.key_alt(rep_alt) for rep_alt in alt_dir]
            
            for i in range(len(alt_dir)):
                
                path_data = sorted(os.listdir(os.path.join(self.path,alt_dir[i])))
                if len(path_data)>=14:
                    path_data = path_data[1:]
                
                path_z.append(os.path.join(self.path,alt_dir[i]))
                data_z.append(sorted(path_data, key=hlp.key_data))
 
        choice  = [ " " , "les differentes altitudes sont : {}".format(self.z)][verbose]
        print(choice)
        # extraction des donnees
        for traj in data_z[self.z.index(z)]:
            data = pd.read_csv(os.path.join(path_z[self.z.index(z)], traj))
            data.head()
            data = data.iloc[2:, :6]
            data.drop_duplicates(inplace=True)
            data = np.array(data, dtype=np.float64)
            all_data.append(data)
            
        return all_data
    
    def extract_dipole_value_traji(self,  num_traj_list, z): # [ [ dp1, dp2 ...dp13], ...........[ dp1, dp2 ...dp13], ]

        def echantillonnage(values, min) :
            
            values_ech = []
            for data_traj in values :
                list_dp = []
                for data_dp in data_traj :
                    list_dp.append(np.concatenate((data_dp[1:min,0].reshape(-1,1), data_dp[1:min,5].reshape(-1,1)), axis=1)) # or df.drop
                values_ech.append(list_dp)
            return values_ech
        
        def cutting(values, x_val=np.linspace(40, 160,300)):
            
            essais1 = True
            
            values_cut = []
            for data_traj in values :
                list_dp = []
                for data_dp in data_traj :
                    tsec = data_dp[:, 0]*1e-3
                    
                    if essais1 :
                        dist_y = (tsec - tsec[0])*self.vx
                        y_val = np.interp(x_val,dist_y, data_dp[:,5])
                    
                    else :
                        dist_y = (tsec - tsec[0])*self.vy + 40
                        y_val = np.interp(x_val,dist_y, data_dp[:,5])

                    dp = np.concatenate((x_val.reshape(-1,1),y_val.reshape(-1,1)), axis =1)
                    list_dp.append(dp)
                    
                values_cut.append(list_dp)
            return values_cut
                
        traj_dipole_value = []
        all_data = self.extract_data_frome_file(z, verbose=False)
        
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
        
        # --------echantillonnage
        min_signal_shape = min(signal_shape)
        
        print(" min ", min_signal_shape)
        # print(" min signal shape = " , min_signal_shape)
        # traj_dipole_value = echantillonnage(traj_dipole_value, min_signal_shape)        
        
        # --------interpolation
        x_val = [ np.linspace(40,460,1064), np.linspace(40,160,300)] [self.TEST -1 ]
        
        traj_dipole_value = cutting(traj_dipole_value, x_val)
        #---------------------
        
        return traj_dipole_value

    
    def save_data_z(self, z = 5):
        
        traj_dipole_value = self.extract_dipole_value_traji(range(self.traj), z=z)
        num_traj = len(traj_dipole_value)
        
        num_dipole = 13
        signal_shape = traj_dipole_value[0][0].shape[0]-1
        
        print("signal shape  : ", signal_shape)
        print("Nombre de trajectoire   : ", num_traj)
        
        X = np.zeros((num_traj,num_dipole*signal_shape ))
        
        for i, data_traj in enumerate(traj_dipole_value) :
            dp = np.array([])
            for data_dp in data_traj :
                _,integrale_derivee_I = hlp.integrale_derivee(data_dp[:,0],data_dp[:,1])               
                dp = np.concatenate((dp, integrale_derivee_I))
            X[i] = dp.ravel()
        
        
        # y  =  np.array([0]*5 + [1]*8 + [0]*4).reshape((num_traj,1))
        if self.TEST == 1 :
            y = np.array([0]*5 + [1]*8 + [0]*4).reshape((num_traj,1))
        elif self.TEST == 2 :
            y = np.array([0]*5 + [1]*8 + [0]*4).reshape((num_traj,1))
            
        
        X = np.concatenate((X, np.array([signal_shape]*num_traj).reshape(-1,1)), axis=1) # ajout de la valeur d'echantiollonnage
        X = np.concatenate((X, y), axis=1) # ajout des labels
        
        print(X.shape)
        
        df = pd.DataFrame(X)       
        if not os.path.exists(self.path_to_data_dir) :
            os.mkdir(self.path_to_data_dir)
            
        df.to_csv(os.path.join(self.path_to_data_dir, "data{}.csv".format(z)), header=False, index=False)
        


    def interpolation_courbe(self, list_signal,  alt_z , alt_z_val):

        signal_allz = []        
        for i in range(list_signal[0].shape[0]) :
            I_z = []
            for signal in list_signal :
                I_z.append(signal[i])        
            I_allz = np.interp(alt_z_val, alt_z, I_z)
            signal_allz.append(I_allz)

        return np.array(signal_allz, dtype=np.float64).T # signal pour toutes les altitudes 
    
    
    def extract_dipole_value_all_altitude(self):

        data = []
        X_no_interp = []
        
        for z in self.z :
            data.append(pd.read_csv(os.path.join(self.path_to_data_dir, 'data{}.csv'.format(z)), header=None, index_col=None))


        signal_shape_array = data[0].iloc[:,-2] # array  of signal_shape
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]
        self.dp_n = int(data[0].shape[1]/ self.signal_shape) # nombre de dipole
        self.traj_num = int( data[0].shape[0] )


        for i in range(len(self.z )):
            X_no_interp.append(np.array(data[i].iloc[:,:-2], dtype=np.float64).reshape(-1,self.dp_n,self.signal_shape))
       
        pas = 0.5
        alt_z_val = np.arange(5, 20+pas, pas) 
        alt_z = [5 , 10 , 15, 20]
        self.num_alt = len(alt_z_val)
        X_interp = np.zeros((self.num_alt, self.traj_num, self.dp_n, self.signal_shape))
        y = np.arange(5, 20+pas, pas).reshape(-1,1) # array  of signal_shape
        
        # #     #------------------------features + window------------------      
        print(" le nombre de trajectoires est {} ".format(self.traj_num))    
        print(" le nombre de dipoles est {} ".format(self.dp_n))  
        print(" nombre d'altitude obtenue par regression {}".format(self.num_alt))
    
        for k in range(len(alt_z_val)):
            for i in range(self.traj_num):
                for j in range(self.dp_n):
                    list_signal = []
                    for x in X_no_interp :
                        list_signal.append(x[i][j])
                        
                    X_interp[k][i][j] = self.interpolation_courbe(list_signal, alt_z,  alt_z_val)[k]


        X_interp = X_interp.reshape(-1,self.traj_num*self.dp_n*self.signal_shape)
        X_interp = np.concatenate((X_interp, np.array([self.signal_shape]*self.num_alt ).reshape(-1,1)), axis=1)
        X_interp = np.concatenate((X_interp, y), axis=1)

        df = pd.DataFrame(X_interp)

        if not os.path.exists(self.path_to_data_dir) :
            os.mkdir(self.path_to_data_dir)
            
        df.to_csv(os.path.join(self.path_to_data_dir, "data_all_z.csv"), header=False, index=False)       


        # fig, axes = plt.subplots(3,2, figsize=(15,12))
        # axes = axes.flatten()
        
        # for j in range(6):
        #     axes[j].plot(X_interp[0,j*self.signal_shape:(j+1)*self.signal_shape])
            
        print( "file for all altitude created")


        
    def plot_dipole_traji_dipolej(self, num_traj_list, num_dipole_list, z = 5, pipe = None, axis_x = True):

        
        traj_dipole_value = self.extract_dipole_value_traji(num_traj_list, z=z, pipe=pipe)
        
        for traj_num,list_dipole in enumerate(traj_dipole_value) :
            fig = plt.figure()
            for i in num_dipole_list:
                
                dp = list_dipole[i]
                
                # if essais = 1
                x = dp[:, 0] 
                #     tsec = range(dp.shape[0])
                #     x = (tsec, x)[axis_x]
                
                # tsec  = ( dp[:, 0] - dp[0, 0] )*1e-3
                # x = tsec * self.vy + 40
                tsec =range(100)
                
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

        plt.show()
  


#%% define a box
if __name__ == '__main__':

    data_extraction = Data_extraction(ESSAI = 1, TEST=2)
    
    # interpolation
    for z in range(5,25,5):
        data_extraction.save_data_z(z=z)
    data_extraction.extract_dipole_value_all_altitude()
        
plt.show()
    # data_extraction = Data_extraction(ESSAIS = 2, TEST=1)
    
    
    # data_extraction.extract_data_frome_file(z = 5)
    # data_extraction.plot_dipole_traji_dipolej([7], range(13), z = 4)
    # # data_extraction.features_labels_save(pipe=1)
    
    # # data_extraction.save_data_in_file(z=20) # Z = 5, 10 ,15, 20
    # # data_extraction.extract_dipole_value_traji(range(17),  z=5)
    
    # # Traitement donnees
    # data_extraction.plot_dipole_traji_dipolej(range(1), range(1),  z=5, axis_x=True, pipe=1)
    # plt.show()
    
    # z = 20
    # data_extraction.plot_dipole_traji_dipolej(range(4,9), range(1),  z=20)
    
    # labelisation
    # data_extraction.plot_dipole_traji_dipolej(range(17), range(8),  z=15)
    
    # else
    # data_extraction.plot_cartographie([5], z=5) # les dipoles selectionnes
    # data_extraction.plot_altitude_echosondeur(range(1))
    # data_extraction.plot_dipole_traji_dipolej_echosondeur([1,2,4],[1,4,5])
    
#%%


