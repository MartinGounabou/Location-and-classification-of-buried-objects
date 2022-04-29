import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sys 


def prediction() :
    
    
    model_filename = "model/LR_model.sav"
    
    ## ----------------
    data = pd.read_csv("model/test_data.csv")
    X_test = np.array(data.iloc[:,:-1])
    y_pred = np.array(data.iloc[:,-1])
    ## ----------------

    model = pickle.load(open(model_filename, 'rb'))

    y_pred_load_model = model.predict(X_test[10,:].reshape(1,130))
    print(y_pred_load_model)
    
    for i in range(13) :
        plt.figure()
        plt.bar(range(130), X_test[10,i:i+10])
    
    plt.show()
    
if __name__ == '__main__':
    prediction()