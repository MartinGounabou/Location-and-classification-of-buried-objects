from tkinter import *
from lib_ihm import Data_extraction
import  matplotlib.pyplot as plt 


class IHM(Data_extraction):
    
    def __init__(self) -> None:
        super().__init__()
        self.fenetre = Tk()
        self.fenetre.title("IHM")

        # Personnaliser la couleur de l'arrière-plan de la fenêtre principale :
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
        bouton1 = Button (self.fenetre, text = "Display", command=self.plot_graphe1)
        
        
        texte1.place(bordermode=OUTSIDE, height=40, width=400)
        bouton1.place(x = 100,y =360,bordermode=OUTSIDE, height=60, width=200)
        
        
        OptionList_traj = range(1,18)
        OptionList_alt = range(5,25,5)
        OptionList_pipe = range(1,4)
        OptionList_dipole = self.dipole
        
        self.variable = StringVar(cadre1)
        self.variable2 = StringVar(cadre2)
        self.variable3 = StringVar(cadre1)
        self.variable4 = StringVar(cadre2)  
           
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

        # fenetre.mainloop()
        
    def plot_graphe1(self):

        self.plot_dipole_traji_dipolej(num_traj_list=[int(self.variable3.get())-1], 
                                       num_dipole_list=[self.dipole.index(int(self.variable4.get()))],
                                    z = int(self.variable2.get()), pipe=int(self.variable.get()))

        # plt.ion()
        # print("plotting")
        plt.ion()
        plt.show()
 
        return None                     
    
    def plot_graphe2(self):
    
        self.plot_dipole_traji_dipolej(num_traj_list=[int(self.variable3.get())-1], 
                                    num_dipole_list=range(13),
                                    z = int(self.variable2.get()), pipe=int(self.variable.get()))

        plt.ion()
        plt.show()
        return None       

if __name__ == '__main__' :
    
    ihm = IHM()
    ihm.fenetre.mainloop()


