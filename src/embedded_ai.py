import os 
import pandas as pd
import numpy as np
import pickle

def key(file):

    split_ =  file.split('_')
    n_data=  split_[1].split('a')
    
    return int(n_data[2]) 

def list_file(path):
    
    list = sorted([file for file in os.listdir(path) if file.startswith(fileExt)], 
            key=key)
    return  list

def current(data):
    # A = np.empty((0,10))A = np.empty((0,10))
    
    traj_dipole_value = []
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
            
    traj_dipole_value.append(
                [dp12, dp13, dp16, dp18, dp26, dp27, dp28, dp36, dp37, dp38, dp45, dp58, dp68])
 
    return traj_dipole_value
        
if __name__ == '__main__' :
    
    # path = "/home/root/logs"
    path = "logs_test_embarques" # verifier les droits 
    fileExt = "logs_data"
    model_filename = "model/ET_model.sav"
    model = pickle.load(open(model_filename, 'rb'))
    
    
    n0_file = len(list_file(path))
    
    while True :

        n_file = len(list_file(path))
        
        if  n_file == n0_file : # or = n0+1
            n0_file = n_file
            
            data = pd.read_csv(os.path.join(path, list_file(path)[-1]), on_bad_lines='skip', dtype='unicode')
            data = data.iloc[2:, :6]
            data.drop_duplicates(inplace=True)
            data = np.array(data, dtype=np.float64)

            features = current(data)
            
            features = data.iloc[-1,:]
            
            #prediction et logs, try sur cette func
            # alt = model.predict(np.array(features).reshape(1,130))
            
           
            
































































            

    # charger mon modèle
    
    # files = [os.path.join(fileDir, _) for _ in os.listdir(fileDir) if _.endswith(fileExt)]
    
    
    # files = sorted([os.path.join(path, file)  for file in os.listdir(path) if file.startswith(fileExt)], 
    #                key= lambda item : int(item.split('_')[1].split('a')[2] )) #select file begin by logs_data and sort
    
    