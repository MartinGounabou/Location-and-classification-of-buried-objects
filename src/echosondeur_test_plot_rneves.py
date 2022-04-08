

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


mypath = '/home/martinho/Documents/Electric_sense_for_burried_objects_locating_classification/Datasets/ESSAIS/3BO-COND/Eau_salee/TEST2/Pipe1(BO=1cm)/Echosondeur/Test2-Pipe1(BO=1cm)'
        
# vx = 7.8780478 # [cm/s] à remesurer
# vx = 3.176218897
vy = 9.3622333 # cm/s

vx = 4

# files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith('Traj')]

altitude = []
heure = []
minu = []
sec = []
misec = []
tsec = []
px = []




files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith('Traj')]

print(files)
for f in sorted(files):
    
    data_traj = pd.read_csv(mypath + "/" + str(f) , delimiter=',', encoding = "ISO-8859-1", engine='python',skiprows=2,header = None,index_col=False)
    
    altitude.append(np.array(data_traj[4]))
    heure.append(np.array(data_traj[0]))
    minu.append(np.array(data_traj[1]))
    sec.append(np.array(data_traj[2]))
    misec.append(np.array(data_traj[3]))
    
for i in range(len(files)):
    tsec.append([heure[i][f]*3600 + minu[i][f]*60  + sec[i][f] + misec[i][f]/1000 for f in range(len(altitude[i]))])
    


ech_cut0 = [129,166,143,132,161,153,177,128,120,121,122,134,132,130,124,122,122]
ech_cutf = [477,515,496,480,507,505,530,477,468,482,471,495,481,479,474,488,471]

# Position en x

for i in range(len(altitude)):
    t0 = tsec[i][ech_cut0[i]]      
    px.append((tsec[i][ech_cut0[i]:ech_cutf[i]] - t0)*vx)


traj = [0,1,2,3,4,5,6]
plt.figure(dpi=200)
plt.title('Test 2 - Pipe 1 - Distance entre l''echosondeur et le sol', fontsize = 14)

for i in traj:
    # plt.plot(altitude[0][i][ech_cut0[i]:ech_cutf[i]]/10 - 7, linewidth = 0.2)
    plt.plot(px[i] + 14.5, altitude[i][ech_cut0[i]:ech_cutf[i]]/10 - 7, linewidth = 0.2)

    
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Y [cm]')
plt.ylabel('Altitude [cm]',fontsize=12)
leg = plt.legend(np.array(traj)+1, loc = 'best')
# # leg = plt.legend(np.arange(0,14)+1)
# # set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.0)
plt.show()