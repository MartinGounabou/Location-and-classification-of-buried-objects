# echo-client.py

import socket
import numpy as np


HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 6532  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True :
        data = s.recv(1024)
        
        data = np.array(data.decode())
        print(f"Received {data!r}")