#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:12:29 2022

@author: rneves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import percentile_filter
import copy
import matplotlib.ticker as mtick
from scipy.integrate import cumtrapz  # Module d'intégration cumulé "cumtrapz"
from os import listdir
from os.path import isfile, join
import os 

# Efface les variables en memoire
# %reset -f

#
# headRowsSkip = 5
# numberRows = 6804 # multiple de 28
# filenumber = 13
# filenumberstep = 10
# datasetnumber1 = 0
# dipolenumber = 1

# Colonnes dans les fichiers de log
EE = 1
ER = 2
Urms = 3
Irms = 5
RTC = 0

vy = 4

x0 = 75 # position x initial du test
y0 = 14.5 # position y initial du test 

impedance = []
courant = []
rtc = []

courant2 = []
rtc2 = []

# Dipôle à analiser
ee = str(1)
er = str(2)
dipo = ee + er

freq = 25
alt = 5

dips = ['13', '16', '18', '36', '38','12','26','37','28','58', '68', '27', '45']

courants = []
rtcs = []
impedances = []
pos = []

for i in range(len(dips)):
    courants.append([])
    rtcs.append([])
    impedances.append([])
    pos.append([])

i_dip = dips.index(dipo) #indice du dipôle


# mypath = '/home/rneves/Documents/Essais/New BO - skid Forssea/Eau salée/Test 2/Pipe 1 - BO = 1 cm/5 cm/'
mypath = os.path.abspath("../Datasets/ESSAIS 1/3BO-COND/Eau_salee/TEST 2/Pipe 1 - BO = 1 cm/5 cm/")


files = [f for f in listdir(mypath) if isfile(join(mypath, f))] # List all files
tn = [0,5,10,15,17,19,21,23,25,27,29,31,33,35,40,45,50]
# files = sorted(files)

# Lecture du fichier de mesure
# for datasetNumber in range(0,filenumber):
for data in sorted(listdir(mypath)):
    data_traj = pd.read_csv( join(mypath, data), delimiter=',', encoding = "ISO-8859-1", engine='python',header=None, skiprows=3)


    # Récupération des mesures par dipôle
    EE1ER2 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==2)]
    EE1ER3 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==3)]
    EE1ER4 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==4)]
    EE1ER5 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==5)]
    EE1ER6 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==6)]
    EE1ER7 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==7)]
    EE1ER8 = data_traj[(data_traj[EE]==1) & (data_traj[ER]==8)]

    EE2ER3 = data_traj[(data_traj[EE]==2) & (data_traj[ER]==3)]
    EE2ER4 = data_traj[(data_traj[EE]==2) & (data_traj[ER]==4)]
    EE2ER5 = data_traj[(data_traj[EE]==2) & (data_traj[ER]==5)]
    EE2ER6 = data_traj[(data_traj[EE]==2) & (data_traj[ER]==6)]
    EE2ER7 = data_traj[(data_traj[EE]==2) & (data_traj[ER]==7)]
    EE2ER8 = data_traj[(data_traj[EE]==2) & (data_traj[ER]==8)]

    EE3ER4 = data_traj[(data_traj[EE]==3) & (data_traj[ER]==4)]
    EE3ER5 = data_traj[(data_traj[EE]==3) & (data_traj[ER]==5)]
    EE3ER6 = data_traj[(data_traj[EE]==3) & (data_traj[ER]==6)]
    EE3ER7 = data_traj[(data_traj[EE]==3) & (data_traj[ER]==7)]
    EE3ER8 = data_traj[(data_traj[EE]==3) & (data_traj[ER]==8)]

    EE4ER5 = data_traj[(data_traj[EE]==4) & (data_traj[ER]==5)]
    EE4ER6 = data_traj[(data_traj[EE]==4) & (data_traj[ER]==6)]
    EE4ER7 = data_traj[(data_traj[EE]==4) & (data_traj[ER]==7)]
    EE4ER8 = data_traj[(data_traj[EE]==4) & (data_traj[ER]==8)]

    EE5ER6 = data_traj[(data_traj[EE]==5) & (data_traj[ER]==6)]
    EE5ER7 = data_traj[(data_traj[EE]==5) & (data_traj[ER]==7)]
    EE5ER8 = data_traj[(data_traj[EE]==5) & (data_traj[ER]==8)]

    EE6ER7 = data_traj[(data_traj[EE]==6) & (data_traj[ER]==7)]
    EE6ER8 = data_traj[(data_traj[EE]==6) & (data_traj[ER]==8)]

    EE7ER8 = data_traj[(data_traj[EE]==7) & (data_traj[ER]==8)]
    
    # Calcul d'impédance par dipôle
    impedance12 = EE1ER2[Urms]/EE1ER2[Irms]
    impedance13 = EE1ER3[Urms]/EE1ER3[Irms]
    impedance14 = EE1ER4[Urms]/EE1ER4[Irms]
    impedance15 = EE1ER5[Urms]/EE1ER5[Irms]
    impedance16 = EE1ER6[Urms]/EE1ER6[Irms]
    impedance17 = EE1ER7[Urms]/EE1ER7[Irms]
    impedance18 = EE1ER8[Urms]/EE1ER8[Irms]
    
    impedance23 = EE2ER3[Urms]/EE2ER3[Irms]
    impedance24 = EE2ER4[Urms]/EE2ER4[Irms]
    impedance25 = EE2ER5[Urms]/EE2ER5[Irms]
    impedance26 = EE2ER6[Urms]/EE2ER6[Irms]
    impedance27 = EE2ER7[Urms]/EE2ER7[Irms]
    impedance28 = EE2ER8[Urms]/EE2ER8[Irms]
    
    impedance34 = EE3ER4[Urms]/EE3ER4[Irms]
    impedance35 = EE3ER5[Urms]/EE3ER5[Irms]
    impedance36 = EE3ER6[Urms]/EE3ER6[Irms]
    impedance37 = EE3ER7[Urms]/EE3ER7[Irms]
    impedance38 = EE3ER8[Urms]/EE3ER8[Irms]
    
    impedance45 = EE4ER5[Urms]/EE4ER5[Irms]
    impedance46 = EE4ER6[Urms]/EE4ER6[Irms]
    impedance47 = EE4ER7[Urms]/EE4ER7[Irms]
    impedance48 = EE4ER8[Urms]/EE4ER8[Irms]
    
    impedance56 = EE5ER6[Urms]/EE5ER6[Irms]
    impedance57 = EE5ER7[Urms]/EE5ER7[Irms]
    impedance58 = EE5ER8[Urms]/EE5ER8[Irms]
    
    impedance67 = EE6ER7[Urms]/EE6ER7[Irms]
    impedance68 = EE6ER8[Urms]/EE6ER8[Irms]

    impedance78 = EE7ER8[Urms]/EE7ER8[Irms]
    
    
    # impedance.append(np.array(impedance13))
    impedance.append(np.array(globals()['impedance' + ee + er]))
    # courant[:, datasetNumber] = EE1ER3[Irms]/np.amax(EE1ER3[Irms])
    # courant.append(np.array(EE1ER3[Irms]))
    courant.append(np.array(globals()['EE' + ee + 'ER' + er][Irms]))
    # rtc.append(np.array(EE1ER3[RTC]))
    rtc.append(np.array(globals()['EE' + ee + 'ER' + er][RTC]))
    
    for i in range(len(dips)):
        courants[i].append(np.array(globals()['EE' + dips[i][0] + 'ER' + dips[i][1]][Irms]))
        rtcs[i].append(np.array(globals()['EE' + dips[i][0] + 'ER' + dips[i][1]][RTC]))
        impedances[i].append(np.array(globals()['impedance' + dips[i][0] + dips[i][1]]))

# Supprimer des données avec RTC répété 

ind_rep = []
for d in range(len(dips)):
    ind_rep.append([])
    for i in range(len(rtcs[d])):
        ind_rep[d].append([])
        for j in range(1,len(rtcs[d][i])):
            if rtcs[d][i][j] == rtcs[d][i][j-1]:
                ind_rep[d][i].append(j)
        
for d in range(len(dips)):
    for i in range(len(rtcs[d])):
        courants[d][i] = np.delete(courants[d][i],ind_rep[d][i])
        rtcs[d][i] = np.delete(rtcs[d][i],ind_rep[d][i])
        impedances[d][i] = np.delete(impedances[d][i],ind_rep[d][i])
        
# Position en y
for d in range(len(dips)):
    for i in range(len(rtcs[d])):
        p0 = rtcs[d][i][0]
        pos[d].append((rtcs[d][i] - p0)/1000*vy)
        
        
'''

#%% Interpolation courants

# Axe x
# Dans tous les trajectoires, pour chaque dipôle en dips, la position finale minimale atteinte
min_pos_dips = [min([pos[d][i][-1] for i in range(len(pos[d]))]) for d in range(len(dips))]
pos_min = min(min_pos_dips) # le minimum en considérant tous les dipôles en dips

# y_int = np.linspace(0,pos_min,300) # positions en x pour l'interpolation
y_int = np.linspace(40,160,300) # positions en x pour l'interpolation
courants_int = []

for d in range(len(dips)):
    courants_int.append([])
    for i in range(len(courants[d])):
        courants_int[d].append(np.interp(y_int,pos[d][i]+y0,courants[d][i]))
        # courants_int[d].append((np.interp(y_int,pos[d][i],courants[d][i]))/max(courants[d][i]))
  
#%% Cartopraphie avec courant interpolé et countourf
i_dip = dips.index('27') #indice du dipôle

for i  in range(13) :
    i_dip = i
    traj = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    px = np.array(tn) + x0

    cint = []
    for i in traj:
        # cint.append(courants_int[i_dip][i]/max(courants[d][i]))
        cint.append(courants_int[i_dip][i])

    #% Plot cartographie 
    [X,Y] = np.meshgrid(y_int,np.array(px)) 
    plt.figure(dpi=200)
    plt.title('I dip' + dips[i_dip] + ' - Test 1 - skid Forssea, HO = ' + str(alt) + ' cm')
    plt.contourf(X,Y,np.array(cint),cmap = 'jet', levels = 100)
    plt.xlabel('Y(cm)')
    plt.ylabel('X(cm)')
    fmt_sci = '%0.5f'
    yticks = mtick.FormatStrFormatter(fmt_sci)
    # plt.xticks(ticks=np.arange(60, 160, 20))
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.tight_layout()
# plt.gca().invert_xaxis()
plt.show()

'''
#%% Int(dI/dx)

#Calcul de la dérivée
nb_pts_derivee = 5

def derivee(y, x, n):
    d = []
    for i in range(0, len(y)-n):
        dy = y[i+n]-y[i]
        dx = x[i+n]-x[i]
        d.append(dy/dx)
    return d

# Initialization du vecteur
di_dx = []
for i in range(len(dips)):
    di_dx.append([])

for d in range(len(dips)):
    for i in range(len(courants[d])):
        di_dx[d].append(derivee(courants[d][i],pos[d][i],nb_pts_derivee))
        

# Intégration du dI/dX

integ_di_dx = []
for i in range(len(dips)):
    integ_di_dx.append([])

p_integ = []
for d in range(len(dips)):
    p_integ.append([])
    for i in range(len(di_dx[d])):
        integ_di_dx[d].append(cumtrapz(np.array(di_dx[d][i]),pos[d][i][0:len(di_dx[d][i])],dx=1.0, axis=-1, initial=None))
        p_integ[d].append(pos[d][i][0:len(integ_di_dx[d][i])])


#%% Interpolation intégrales 

# Axe y
# Dans tous les trajectoires, pour chaque dipôle en dips, la position finale minimale atteinte
min_int_dips = [min([p_integ[d][i][-1] for i in range(len(p_integ[d]))]) for d in range(len(dips))]
pos_int_min = min(min_int_dips) # le minimum en considérant tous les dipôles en dips

# y_integ_int = np.linspace(0,pos_int_min,1000) # positions en x pour l'interpolation
y_integ_int = np.linspace(40,160,300) # positions en x pour l'interpolation
integ_di_dx_int = []

for d in range(len(dips)):
    integ_di_dx_int.append([])
    for i in range(len(integ_di_dx[d])):
        integ_di_dx_int[d].append(np.interp(y_integ_int,p_integ[d][i]+y0,integ_di_dx[d][i]))


#%% Cartopraphie avec int(dI/dx) interpolé et countourf
i_dip = dips.index('12') #indice du dipôle


traj = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# px = [10*traj[i]+40 for i in range(len(traj))]
px = np.array(tn) +x0

Idint = []
for i in traj:
    Idint.append(integ_di_dx_int[i_dip][i])

#% Plot cartographie 
[X,Y] = np.meshgrid(y_integ_int,px) 
plt.figure(dpi=200)
plt.title('Int(dI/dx) dip ' + dips[i_dip] + ' - Test 2 (pipe 1) - Alt = ' + str(alt) + ' cm')
plt.contourf(X,Y,np.array(Idint),cmap='jet',levels = 200)
plt.xticks(ticks=np.arange(60, 170, 30))
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
fmt_sci = '%0.5f'
yticks = mtick.FormatStrFormatter(fmt_sci)
plt.gca().set_aspect("equal")
plt.colorbar()
plt.tight_layout()
# plt.gca().invert_xaxis()
plt.show()

