
from tkinter import *

from data_manipulation_burried_object_localisation import Data_extraction
import  matplotlib.pyplot as plt 
import pandas  as pd 
import numpy as np
import utils.helper as hlp

import os 

class IHM(Data_extraction):
    
    def __init__(self) -> None:
       
        super().__init__(ESSAI=2, TEST=2)
        self.path_to_data = os.path.join(self.path_to_data_dir, 'data_all_z.csv')

        self.z = [ 4 , 6, 8, 10, 12]

    
        print( " end loading data")
        self.fenetre = Tk()
        self.fenetre.title("IHM")

       
        
        # fenetre.resizable(False, False)

        # Personnaliser la couleur de l'arrière-plan de la fenêtre principale
        self.fenetre.config(bg = "#87CEEB")
        self.fenetre.geometry("400x500")


        # Création d'une fenêtre avec la classe Tk :
        # Création d'une barre de menu dans la fenêtre :
        menu1 = Menu(self.fenetre)
        menu1.add_cascade(label="Fichier")
        menu1.add_cascade(label="Options")
        menu1.add_cascade(label="Aide")
        
        
        # Configuration du menu dans la fenêtre
        self.fenetre.config(menu = menu1)
        # Affichage de la fenêtre créée :
        
        cadre1 = Frame(self.fenetre)
        cadre1.place(x = 0,y =150,bordermode=OUTSIDE, height=40, width=100)
        cadre2 = Frame(self.fenetre)
        cadre2.place(x = 100,y =150,bordermode=OUTSIDE, height=40, width=100)
        cadre3 = Frame(self.fenetre)
        cadre3.place(x = 200,y =150,bordermode=OUTSIDE, height=40, width=100)
        cadre4 = Frame(self.fenetre)
        cadre4.place(x = 300,y =150,bordermode=OUTSIDE, height=40, width=100)
        
        
        texte1 = Label (self.fenetre, text= "affichage de graphe")

        bouton1 = Button (self.fenetre, text = "Display1", command=self.plot1)
        bouton2 = Button (self.fenetre, text = "Display2", command=self.plot2)
        
        
        texte1.place(bordermode=OUTSIDE, height=40, width=400)
        bouton1.place(x = 50,y =360,bordermode=OUTSIDE, height=60, width=100)
        bouton2.place(x = 250,y =360,bordermode=OUTSIDE, height=60, width=100)
        
        
        OptionList_pipe = range(1,4)
        OptionList_alt = self.z
        OptionList_traj = range(1,18)
        OptionList_dipole =  self.dipole
        
        self.variable = StringVar(cadre1)
        self.variable2 = StringVar(cadre2)
        self.variable3 = StringVar(cadre3)
        self.variable4 = StringVar(cadre4)  
           
        self.variable.set("pipe")
        self.variable2.set("alt")
        self.variable3.set("traj")
        self.variable4.set("dipole") 
        
              
        opt = OptionMenu(cadre1, self.variable, *OptionList_pipe)
        opt.config(width=30,height=40, font=('Helvetica', 12))
        opt.pack(side=LEFT)
        
        opt2 = OptionMenu(cadre2, self.variable2, *OptionList_alt)
        opt2.config(width=30,height=40, font=('Helvetica', 12))
        opt2.pack(side=LEFT)

        opt3 = OptionMenu(cadre3, self.variable3, *OptionList_traj)
        opt3.config(width=30,height=40, font=('Helvetica', 12))
        opt3.pack(side=LEFT)
        
        
        opt4 = OptionMenu(cadre4, self.variable4, *OptionList_dipole)
        opt4.config(width=30,height=40, font=('Helvetica', 12))
        opt4.pack(side=LEFT)



    def read(self, num_z):
        
        data = pd.read_csv(self.path_to_data , header=None, index_col=None)

        self.X = data.iloc[:,:-2]
        self.y = data.iloc[:,-1] # array  of signal_shape
        signal_shape_array = data.iloc[:,-2] # array  of signal_shape
        self.X = np.array(self.X, dtype=np.float64)
        self.y = np.array(self.y, dtype=np.float64)
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]
    
        # # #     #------------------------features + window------------------
        
        self.traj_num = 17
        self.alt = self.X.shape
        self.dp_n = 13 # nombre de dipole

        print(self.X.shape)
        self.X = (self.X[num_z]).reshape(self.traj_num, self.dp_n, self.signal_shape)

    def plot1(self):


        traj =  int(self.variable3.get())-1
        i = self.dipole.index(int(self.variable4.get()))
        z = float(self.variable2.get())


        print( z, traj, i)
        self.read(self.z.index(z))


        fig , ax = plt.subplots(1,2, figsize=(10,8))
        ax = ax.flatten()

        Y = list(self.X[traj][i])
        lissage_dp = hlp.lissage(Y, 20)


        ax[0].plot(np.linspace(40,160,300)[:-1],lissage_dp,  label="lissage_dp{}".format(self.dipole[i]), linewidth=1)
        print("plotting")
        plt.grid()

        self.plot_real_z(ax, [traj], [i], z =z )

        plt.ion()
        plt.show()

    def plot2(self):


        traj =  int(self.variable3.get())-1
        z = float(self.variable2.get())

        self.read(self.z.index(z))


        fig , ax = plt.subplots(1,2, figsize=(10,8))
        ax = ax.flatten()

        for i in range(13) :
            Y = list(self.X[traj][i])
            lissage_dp = hlp.lissage(Y, 20)
            ax[0].plot(np.linspace(40,160,300)[:-1],lissage_dp,  label="lissage_dp{}".format(self.dipole[i]), linewidth=1)


        self.plot_real_z(ax, [traj], range(13), z =z )

        plt.ion()
        plt.show()



    def plot_real_z(self, ax,  num_traj_list, num_dipole_list, z =5):

        
        traj_dipole_value = self.extract_dipole_value_traji(num_traj_list, z=z)
        
        for traj ,list_dipole in enumerate(traj_dipole_value) :
            for i in num_dipole_list:
                dp = list_dipole[i]
                
                x = dp[:, 0] 

                X, Y = hlp.integrale_derivee(x,dp[:, 1])

                lissage_dp = hlp.lissage(list(Y), 20)
                
                ax[1].plot(X, lissage_dp,  label="lissage_dp{}".format(self.dipole[i]), linewidth=1)


            plt.title(" Traj : {}, alt : {}cm, pipe : {} ".format(num_traj_list[traj]+1, z, self.pipe))
            plt.grid()
          

if __name__ == '__main__' :
    
    ihm = IHM()
    ihm.fenetre.mainloop()