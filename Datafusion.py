import scipy.io as sio
import numpy as np
import pandas as pd


if __name__ =="__main__":
    path=u'DREAMER.mat'
    path_EEG="Extracted_EEG.csv"
    path_ECG="Extracted_ECG.csv"
    data=sio.loadmat(path)
    data_EEG=pd.read_csv(path_EEG).drop(["Unnamed: 0"],axis=1)
    data_ECG=pd.read_csv(path_ECG).drop(["Unnamed: 0"],axis=1)
    a=np.zeros((23,18,3))
    for k in range(0,23):
        for j in range(0,18):
            if data['DREAMER'][0,0]['Data'][0,k]['ScoreValence'][0,0][j,0]<4:
                a[k,j,0]=0
            else:
                a[k,j,0]=1
            if data['DREAMER'][0,0]['Data'][0,k]['ScoreArousal'][0,0][j,0]<4:
                a[k,j,1]=0
            else:
                a[k,j,1]=1
            if data['DREAMER'][0,0]['Data'][0,k]['ScoreDominance'][0,0][j,0]<4:
                a[k,j,2]=0
            else:
                a[k,j,2]=1
    b=pd.DataFrame(a.reshape((23*18,a.shape[2])),columns=['Valence','Arousal','Dominance'])
    feature=pd.concat([data_EEG,data_ECG,b],axis=1)
    print(feature.head())
    feature.to_csv("Feature.csv")