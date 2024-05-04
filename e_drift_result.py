import pandas as pd
import numpy as np
from river import drift
from scipy import stats

def custom_cd(epsilons, drift_index, w_size, confidence, cdd):

    reference_window = epsilons[1500:1600]
    test_window = []
    fp_count = 0
    tp = None

    for idx, epsilon in enumerate(epsilons[1600:]):

        idx = idx+1600


        if len(test_window)==w_size:

            res = cdd(test_window, reference_window)

            if res[1]< confidence:

                if idx>=drift_index:
                    tp = idx
                    return tp, fp_count
                else:
                    fp_count+=1
            
            test_window.pop(0)
            test_window.append(epsilon)
        
        else:
            test_window.append(epsilon)
    return tp, fp_count


def river_cd(epsilons, drift_index, cdd):
    fp_count = 0
    tp = None

    for idx, epsilon in enumerate(epsilons[1500:]):
        idx =idx+1500

        cdd.update(epsilon)

        if cdd.drift_detected:

            if idx >= drift_index:
                tp = idx

                return tp, fp_count
            
            else:
                fp_count+=1
    return tp, fp_count



def e_drift(dataset_name, drift_index):

    #retrieve actual labels and predicted labels
    ddm_tp, ddm_fp, ddm_missed =[], [], []
    eddm_tp, eddm_fp, eddm_missed =[], [], []
    adwin_tp, adwin_fp, adwin_missed =[], [], []
    kswin_tp, kswin_fp, kswin_missed =[], [], []
    mannu_tp, mannu_fp, mannu_missed =[], [], []
    ks_tp, ks_fp, ks_missed =[], [], []


    for i in range(1,6):

        print(i, "next dataset")
        data = pd.read_csv(f"results/{dataset_name}_{i}.csv")
        epsilons = data["epsilon"].to_list()

        #ddm
        ddm_drift = drift.binary.DDM()
        tp, fp = river_cd(epsilons, drift_index, ddm_drift)
        print(tp, fp)
        ddm_fp.append(fp)
        if tp != None:
            ddm_tp.append(tp)
        

        #eddm
        eddm_drift = drift.binary.EDDM()
        tp, fp = river_cd(epsilons, drift_index, eddm_drift)
        print(tp, fp)
        eddm_fp.append(fp)
        if tp != None:
            eddm_tp.append(tp)

        #adwin
        adwin_drift = drift.ADWIN()
        tp, fp = river_cd(epsilons, drift_index, adwin_drift)
        print(tp, fp)
        adwin_fp.append(fp)
        if tp != None:
            adwin_tp.append(tp)
        
        #kswin
        kswin_drift = drift.KSWIN(seed=i)
        tp, fp = river_cd(epsilons, drift_index, kswin_drift)
        print(tp, fp)
        kswin_fp.append(fp)
        if tp != None:
            kswin_tp.append(tp)

        #mannu
        w_size = 100
        confidence = 0.005
        mannu_drift = stats.mannwhitneyu
        tp, fp = custom_cd(epsilons, drift_index, w_size, confidence, mannu_drift)
        print(tp, fp)
        mannu_fp.append(fp)
        if tp != None:
            mannu_tp.append(tp)
        
        #ks
        w_size = 100
        confidence = 0.005
        ks_drift = stats.ks_2samp
        tp, fp = custom_cd(epsilons, drift_index, w_size, confidence, ks_drift)
        print(tp, fp)
        ks_fp.append(fp)
        if tp != None:
            ks_tp.append(tp)
    
    #ddm
    if len(ddm_tp)==0:
        print("ddm: ", 0,0,0)
    else:
        print("ddm", np.mean(ddm_tp), np.std(ddm_tp), np.mean(ddm_fp), np.std(ddm_fp), 5-len(ddm_tp))
    
    #eddm
    if len(eddm_tp)==0:
        print("eddm: ", 0,0,0)
    else:
        print("eddm", np.mean(eddm_tp), np.std(eddm_tp), np.mean(eddm_fp), np.std(eddm_fp), 5-len(eddm_tp))

    #adwin
    if len(adwin_tp)==0:
        print("adwin: ", 0,0,0)
    else:
        print("adwin", np.mean(adwin_tp), np.std(adwin_tp), np.mean(adwin_fp), np.std(adwin_fp), 5-len(adwin_tp))

    #kswin
    if len(kswin_tp)==0:
        print("kswin: ", 0,0,0)
    else:
        print("kswin", np.mean(kswin_tp), np.std(kswin_tp), np.mean(kswin_fp), np.std(kswin_fp), 5-len(kswin_tp))
    
    #mannu
    if len(mannu_tp)==0:
        print("mannu: ", 0,0,0)
    else:
        print("mannu", np.mean(mannu_tp), np.std(mannu_tp), np.mean(mannu_fp), np.std(kswin_fp), 5-len(mannu_fp))
    
    #ks
    if len(ks_tp)==0:
        print("ks: ", 0,0,0)
    else:
        print("ks", np.mean(ks_tp), np.std(ks_tp), np.mean(ks_fp), np.std(ks_fp), 5-len(ks_tp))

def main():

    #params

    dataset_name = "SEA_0_1"
    drift_index = 5000
    
    #3 functions for error-rate, features and 3-drift

    e_drift(dataset_name, drift_index)

if __name__ == '__main__':
    main()
