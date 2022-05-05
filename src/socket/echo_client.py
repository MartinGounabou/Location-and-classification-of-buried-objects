# echo-client.py

import socket
import numpy as np
from use_model import prediction
import os 
import pandas as pd 

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 6532  # The port used by the server


PATH_TO_DIR = os.getcwd()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    cpt = 0 
    input = [ ]
    while True : 
        data = s.recv(1024)
        try :
            data = float(data.decode())
            # print(f"Received {data!r}")
            cpt += 1
            input.append(data)

            if cpt == 130 :
                alt = prediction(input)
                print(" l altitude predit est : ",  alt)
                if not os.path.exists(PATH_TO_DIR):
                    os.mkdir(PATH_TO_DIR)
                
                input.append(alt)
                
                df = pd.DataFrame(input)
                df.to_csv(os.path.join(PATH_TO_DIR,"log_alt.csv"), header=None, index=None)
                
                cpt = 0
                input = []
        except :
            print("erreur de convertion")