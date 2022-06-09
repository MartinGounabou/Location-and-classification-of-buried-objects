# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:39:36 2022

@author: sBausson
"""

from scipy.linalg import  norm
import numpy as np

#########################################################
def Regroupement_dipôles(vbeta, vL, choix=1 ) : 
    ''' identifie les groupes (N>1) de dipôles ayant un paramètre commun 
    1 : beta 
    2 : longueur 
    3 : beta et longueur
    Output : dico[paramètre]= indices 
    '''
    di = dict()
    if choix == 1 :
        print( ' -- Regroupement suivant le paramètre commun : beta ')
        vbr, IC = np.unique( vbeta, return_counts=True )
        vbr = vbr[ IC > 1 ]
        for k in range(len(vbr)) :
            J = np.where( vbeta == vbr[k] )[0]
            di[vbr[k]] = J        
    elif choix == 2 :
        vLr, IC = np.unique( vL, return_counts=True )
        vLr = vLr[ IC > 1 ]
        for k in range(len(vLr)) :
            J = np.where( vL == vLr[k] )[0]
            di[vLr[k]] = J        
    elif choix == 3 :
        print( ' -- Regroupement suivant le paramètre commun : beta et longueur ')
        vbr, IC = np.unique( vbeta, return_counts=True )
        vbr = vbr[ IC > 1 ]
        for k in range(len(vbr)) :
            J = np.where( vbeta == vbr[k] )[0]
            vLr, IC = np.unique( vL[J], return_counts=True )
            vLr = vLr[ IC > 1 ]
            for m in range(len(vLr)) :
                K = np.where( vL[J] == vLr[m] )[0]
                di[ ( vbr[k], vLr[m] ) ] = J[K]
    
    return di

#########################################################
def Calcul_géométrie_dipôles( squid='forssea' ) : 
    ''' à partir de la position des éléctrodes, pour tous les dipôles, 
    calcule : la longueur (int, cm)
              l'anlge beta par rapport à la direction du Squid (int, °)
              la position du centre par rapport au centre du squid (nombre complexe)
            '''
    if squid == 'forssea' : 
        Centre_Electrodes = 5*100*np.asarray( [[0.094,	0.094,	0.094,	0,	    0,	    -0.094,	-0.094,	-0.094],[-0.064,	0,	    0.064,	-0.064,	0.064,	-0.064,	0,	0.064]])
        # Positions des électrodes de 1 à 8, par rapport au centr du Squid
        # dimension en cm
        print(' -- Données géométrique du grand Squid Forssea -- ') 
    
    vL = []
    vname = []
    vbeta = []
    vC = []
    for k in range( Centre_Electrodes.shape[1]-1 ) : 
        for m in range( k+1, Centre_Electrodes.shape[1]) :
            E1 = Centre_Electrodes[:,k]
            E2 = Centre_Electrodes[:,m]
            L  = norm(E1-E2)  # longueur du dipôle 
            iD = (E1-E2)/L    # vecteur unitaire axe dipôle : de E2 vers E1
            C  = (E1+E2)/2    # vecteur de position du centre du dipôle             
            vname.append( 'd%d%d'%(k+1,m+1) )
            vbeta.append( np.arctan(iD[1]/iD[0])*180/np.pi )
            vL.append( L ) 
            vC.append( C[0] + 1j*C[1] ) 
            # print(' -- dipôle %d%d : long=%.1f cm, axe (%.1f,%.1f), centré en (%.1f,%.1f), distance Cd-Cs=%.1f cm, beta=%d°'%(k+1,m+1,L,*iD,*C,norm(DS),np.arctan(iD[1]/iD[0])*180/np.pi ))
            # -- pour le calcul de l'angle alpha dans l'équation du potentiel d'un dipôle parfait, ajouter :
            # DS = -(E1+E2)/2 # vecteur centre dipôle vers centre squid 
            # iD_DS = iD[0]*DS[0] +  iD[1]*DS[1] # produit scalaire iD x DS

    vbeta =  np.asarray( vbeta, dtype='int' )
    vL =  np.asarray(vL, dtype='int' )
    vC =  np.asarray(vC)
    
    return ( vname, vbeta, vL, vC )
    
#########################################################
def Position_Y_diff( vC, di ) :
    ''' Identifie dans les groupes de dipôles (theta et/ou long identique)
    le nbr de position Y diff  '''
    vN = []
    for k in di.keys() : 
        vN.append( len( np.unique( np.imag(vC[di[k]] ) ) ))    
    return np.asarray( vN, dtype=int )



# ------------------------ MAIN  -----------------------------------------
( vname, vbeta, vL, vC ) = Calcul_géométrie_dipôles( squid='forssea' ) 
for choix in (1,3) : 
    di = Regroupement_dipôles(vbeta, vL, choix )
    vN = Position_Y_diff( vC, di )
    print(' Nbr de groupe :', len(di), ' et principaux groupes : ')
    i = -1
    for k in di.keys() : 
        i += 1
        if len(di[k]) >= 1 : 
            if choix == 1 : 
                _ = ''
                for kk in di[k] : 
                    _ += vname[kk]+' '
                # print('  beta = %d °, nbr de dipôles = %d, nbr position Y diff %d : %s'%(k, len(di[k]),vN[i],_[:-2]))
                # print(' %s'%(_[:-2]))
                print(' %s'%(_[:-1]))
                
            elif choix == 3 : 
                _ = ''
                for kk in di[k] : 
                    _ += vname[kk]+' '
                # print('  beta = %d ° et longueur = %d cm, nbr de dipôles = %d, nbr position Y diff %d : %s'%(*k, len(di[k]),vN[i],_[:-2]))
                print(' %s'%(_[:-1]))
                
i = -1                
for k in vname : 
    i += 1
    print(k,' : ',vL[i],' cm')
            
    
    
    