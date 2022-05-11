import shutil
import socket
import random
import time
import numpy as np
import random
import signal
import os


a = 0


def handler(signum, frame):
    global a

    for i in range(a):
        filename2 = "logs_test_embarques\logs_data{}_x19.csv".format(i)
        os.remove(filename2)


signal.signal(signal.SIGINT, handler)

filename1 = "logs_data12_x10.csv"

for i in range(17):

    filename2 = "logs_test_embarques\logs_data{}_x19.csv".format(i)
    shutil.copyfile(filename1, filename2)
    time.sleep(2)
    a += 1

for i in range(17):

    filename2 = "logs_test_embarques\logs_data{}_x19.csv".format(i)
    os.remove(filename2)
