
from email import feedparser
import os
from xml.dom import NO_MODIFICATION_ALLOWED_ERR 
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.datasets import fetch_20newsgroups
from sympy import Segment

def key(file):

    split_ =  file.split('_')
    n_data=  split_[1].split('a')
    
    return int(n_data[2]) 

def list_file(path):
    
    list = sorted([file for file in os.listdir(path) if file.startswith(fileExt)], 
            key=key)
    return  list


def current(data, index):
    
    
    traj_dipole_value = []
    EE = 1
    ER = 2
    
    dip_num = [(1,2), (1,3), (1,6), (1,8),
               (2,6), (2,7), (2,8), (3,6),
               (3,7), (3,8), (4,5), (5,8),
               (6,8)]
    
    # Récupération des mesures par dipôle
    
    for ee, er in dip_num :
        
        dp_current = data[(data[EE]==ee) & (data[ER]==er)][5]
        traj_dipole_value.append(dp_current[10*index:10*(index+1)]) # take 10 values per dp
  
    
    try :
        features = np.array(traj_dipole_value, dtype=np.float64).reshape(1,130)
        EOF = False 
        
    except ValueError :
        EOF = True # probable fin de fichier
        print("probable fin de fichier")
        features = np.arange(130).reshape(1,130)
    
    return features, EOF
        
if __name__ == '__main__' :
    
    # path = "/home/root/logs"
    path_logs = "logs_test_embarques\logs_ia"
    Segment_width  = 10
    path = "logs_test_embarques" # verifier les droits 
    
    fileExt = "logs_data"
    model_filename = "model/ET_model.sav"
    model = pickle.load(open(model_filename, 'rb'))
    data_line = 0
    
    n_init = len(list_file(path)) # nombre de fichier dans le repertoire au debut du test
    n0_file = n_init # numero du trajectoire pointé par le code 
    
    print("n0 = ", n0_file)
    EOF = False
    
    column = [ "dp_{}_{}".format(i,j) 
            for i in range(1,14)  for j in range(1,11)]

    column = column.append("alt")
    

    while True :
        
        n_file = len(list_file(path)) # nombre de fichier dans le document a un instant t
        # index = 0 # remis a zero pour chaque traj
        EOF = False # fin de fichier 
        
        if  n_file != n0_file : # or = n0+1 (n_file != n0_file)
            n0_file = n0_file + 1
            index = 0 
            
            # print("innnnnnnnnnnnnnnnnnnn", not(EOF), not(n0_file == n_file) )
            traj_path = os.path.join(path, list_file(path)[n0_file-1])
            
            print("in 2")
            
            df =  pd.DataFrame(columns=column)
            while not ( EOF and (n0_file!=n_file)) : 
                
          
                data = pd.read_csv(traj_path, delimiter=',',
                                encoding = "ISO-8859-1", engine='python', header=None, 
                                skiprows= 3)

                data.drop_duplicates(inplace=True)
                
                features, EOF = current(data, index)
                
                index = (index+1) if not(EOF)  else index # secion lu dans le fichier ( 10 mesures)
                              
                # print(EOF, n0_file==n_file, n_file)
                
                n_file = len(list_file(path))
                print("traj {} section {} n_file {}".format(n0_file, index, n_file))
                                       
                
                # prediction et logs, try sur cette func
                alt = model.predict(features)
                
                features = np.concatenate((features, alt.reshape(1,1)), axis=1)
        
                df_new_row = pd.DataFrame(features, columns=column)           
                
                df = pd.concat([df,df_new_row], ignore_index=True)
            
            if not os.path.exists(path_logs):
                os.mkdir(path_logs)

            df.to_csv(os.path.join(path_logs,
                    "traj{}.csv".format(n0_file-n_init)), header=False, index=False)
                
                
           
            
































































            

    # charger mon modèle
    
    # files = [os.path.join(fileDir, _) for _ in os.listdir(fileDir) if _.endswith(fileExt)]
    
    
    # files = sorted([os.path.join(path, file)  for file in os.listdir(path) if file.startswith(fileExt)], 
    #                key= lambda item : int(item.split('_')[1].split('a')[2] )) #select file begin by logs_data and sort
    
    