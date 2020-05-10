import scipy.io as sio
import neurokit2 as nk
import pandas as pd
from sklearn import preprocessing as pre
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    total=0
    path=u'DREAMER.mat'
    data=sio.loadmat(path)
    print("ECG signals are being feature extracted...")
    ECG={}
    for k in range(0,23):
        for j in range(0,18):
            basl_l=data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['baseline'][0,0][j,0][:,0]
            stim_l=data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['stimuli'][0,0][j,0][:,0]
            basl_r=data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['baseline'][0,0][j,0][:,1]
            stim_r=data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['stimuli'][0,0][j,0][:,1]
            ecg_signals_b_l,info_b_l=nk.ecg_process(basl_l,sampling_rate=256)
            ecg_signals_s_l,info_s_l=nk.ecg_process(stim_l,sampling_rate=256)
            ecg_signals_b_r,info_b_r=nk.ecg_process(basl_r,sampling_rate=256)
            ecg_signals_s_r,info_s_r=nk.ecg_process(stim_r,sampling_rate=256)
            # processed_ecg_b_l = nk.ecg_intervalrelated(ecg_signals_b_l)
            # processed_ecg_s_l = nk.ecg_intervalrelated(ecg_signals_s_l)
            # processed_ecg_b_r = nk.ecg_intervalrelated(ecg_signals_b_r)
            # processed_ecg_s_r = nk.ecg_intervalrelated(ecg_signals_s_r)
            processed_ecg_l=nk.ecg_intervalrelated(ecg_signals_s_l)/nk.ecg_intervalrelated(ecg_signals_b_l)
            processed_ecg_r=nk.ecg_intervalrelated(ecg_signals_s_r)/nk.ecg_intervalrelated(ecg_signals_b_r)
            processed_ecg=(processed_ecg_l+processed_ecg_r)/2
            if not len(ECG):
                ECG=processed_ecg
            else:
                ECG=pd.concat([ECG,processed_ecg],ignore_index=True)
            total+=1
            print("\rprogress: %d%%" %(total/(23*18)*100),end="")
    # col=ECG.columns.values
    # scaler=pre.StandardScaler()
    # for i in range(len(col)):
    #     ECG[col[i][:-3]] = scaler.fit_transform(ECG[[col[i]]])
    # ECG.drop(col, axis=1, inplace=True)
    # ECG.columns=col
    # print(ECG)
    ECG.to_csv("ECG.csv")