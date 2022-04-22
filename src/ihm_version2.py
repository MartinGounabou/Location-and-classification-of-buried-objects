from binascii import a2b_base64
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
        self.alt_lits = list(np.arange(5,20.5,0.5))
        self.read()
        print( " end loading data")
        fenetre = Tk()
        fenetre.title("IHM")

        fenetre.bind("<Escape>", quit)
        fenetre.bind("x", quit)

        
        # fenetre.resizable(False, False)

        # Personnaliser la couleur de l'arrière-plan de la fenêtre principale
        fenetre.config(bg = "#87CEEB")
        fenetre.geometry("400x500")


        # Création d'une fenêtre avec la classe Tk :
        # Création d'une barre de menu dans la fenêtre :
        menu1 = Menu(fenetre)
        menu1.add_cascade(label="Fichier")
        menu1.add_cascade(label="Options")
        menu1.add_cascade(label="Aide")
        
        
        # Configuration du menu dans la fenêtre
        fenetre.config(menu = menu1)
        # Affichage de la fenêtre créée :
        
        cadre1 = Frame(fenetre)
        cadre1.place(x = 0,y =150,bordermode=OUTSIDE, height=40, width=100)
        cadre2 = Frame(fenetre)
        cadre2.place(x = 100,y =150,bordermode=OUTSIDE, height=40, width=100)
        cadre3 = Frame(fenetre)
        cadre3.place(x = 200,y =150,bordermode=OUTSIDE, height=40, width=100)
        cadre4 = Frame(fenetre)
        cadre4.place(x = 300,y =150,bordermode=OUTSIDE, height=40, width=100)
        
        
        texte1 = Label (fenetre, text= "affichage de graphe")

        bouton1 = Button (fenetre, text = "Display1", command=self.plot_graphe1)
        bouton2 = Button (fenetre, text = "Display2", command=self.plot_graphe2)
        
        
        texte1.place(bordermode=OUTSIDE, height=40, width=400)
        bouton1.place(x = 50,y =360,bordermode=OUTSIDE, height=60, width=100)
        bouton2.place(x = 250,y =360,bordermode=OUTSIDE, height=60, width=100)
        
        
        OptionList_pipe = range(1,4)
        OptionList_alt = self.alt_lits
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

        fenetre.mainloop()
    
    def quit(fenetre):
        fenetre.destroy()

    def read(self, z=5):
        
        data = pd.read_csv(self.path_to_data , header=None, index_col=None)
        self.X = data.iloc[:,:-2]
        self.y = data.iloc[:,-1] # array  of signal_shape
        signal_shape_array = data.iloc[:,-2] # array  of signal_shape
        self.X = np.array(self.X, dtype=np.float64)
        self.y = np.array(self.y, dtype=np.float64)
        self.signal_shape = np.array(signal_shape_array, dtype=np.int64)[0]
    
        # #     #------------------------features + window------------------
        
        self.traj_num = 17
        self.alt = 31
        self.dp_n = 13 # nombre de dipole
        self.X = (self.X).reshape(self.traj_num, self.dp_n, self.signal_shape)

    def plot_graphe1(self):

        plt.figure()

        a = (self.alt_lits).index(float(self.variable2.get()))
        b = int(self.variable3.get())-1
        c =  self.dipole.index(int(self.variable4.get()))

        self.read(z=float(self.variable2.get()))
        
        
        print(float(self.variable2.get()))
        print(a , b , c)

        dp = self.X[b][c]
        x = np.linspace(40,160,299)
        
        # plt.plot(X, Y,  label="dp{}".format(int(self.variable4.get())), linewidth=1)
        plt.plot(x, dp,  label="dp{}".format(int(self.variable4.get())), linewidth=1)


        (plt.xlabel("time(s)"), plt.xlabel("Y(cm)"))
        plt.ylabel("I_rms")
        plt.title(" Traj : {}, alt : {}cm, pipe : {}".format(b, float(self.variable2.get()), 1))
        plt.grid()
        plt.legend()

        plt.ion()
        plt.show()


    def plot_graphe2(self):
        plt.figure()

        a = (self.alt_lits).index(float(self.variable2.get()))
        b = int(self.variable3.get())
        c =  self.dipole.index(int(self.variable4.get()))

        for c in range(0,13) :

            dp = self.X[a][b][c]

            x = np.linspace(40,160,299)
            
            plt.plot(x, dp,  label="dp{}".format(self.dipole[c]), linewidth=1)

            (plt.xlabel("time(s)"), plt.xlabel("Y(cm)"))
            plt.ylabel("I_rms")
            plt.title(" Traj : {}, alt : {}cm, pipe : {}".format(b, float(self.variable2.get()), 1))
            plt.grid()
            plt.legend()

        plt.ion()
        plt.show()

        return None                     
    

if __name__ == '__main__' :
    
    ihm = IHM()