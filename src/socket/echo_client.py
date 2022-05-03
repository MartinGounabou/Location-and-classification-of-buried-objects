# echo-client.py

import socket
import numpy as np
from use_model import prediction


HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 6532  # The port used by the server

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
                y_pred_load_model = prediction(input)
                print(" l altitude predit est : ",  y_pred_load_model)
                cpt = 0
                input = []
        except :
            print("erreur de convertion")
            