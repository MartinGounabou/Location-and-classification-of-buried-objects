import socket
import random
import time
import numpy as np
import random

from scipy import rand

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 6532  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("server en marche")
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")

        while True :
            value  = random.random()
            conn.sendall(str(value).encode())
            time.sleep(0.01)

    
    
    
    