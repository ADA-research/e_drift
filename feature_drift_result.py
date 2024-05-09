import pandas as pd
import numpy as np
from river import drift
from scipy import stats

def custom_cd(features, drift_index, w_size, confidence, cdd_1, cdd_2, cdd_3):

    reference_window_1, reference_window_2, reference_window_3 = [], [], []
    test_window_1, test_window_2, test_window_3 = [], [], []
    fp_count_1, fp_count_2, fp_count_3 = 0,0,0
    tp_1, tp_2, tp_3 = None, None, None

    for idx in range(len(features[1500:1600])):
        idx = idx+1500
        reference_window_1.append(features[idx][0])
        reference_window_2.append(features[idx][1])
        reference_window_3.append(features[idx][2])

    for idx in range(len(features[1600:])):

        idx = idx+1600
        #feature 1
        if len(test_window_1)==w_size:
            res = cdd_1(test_window_1, reference_window_1)

            if res[1]<confidence:
                if idx>=drift_index:
                    tp_1 = idx
                    return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3
                else:
                    fp_count_1+=1
            test_window_1.pop(0)
            test_window_1.append(features[idx][0])
        else:
            test_window_1.append(features[idx][0])
        #feature 2
        if len(test_window_2)==w_size:
            res = cdd_2(test_window_2, reference_window_2)

            if res[1]<confidence:
                if idx>=drift_index:
                    tp_2 = idx
                    return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3
                else:
                    fp_count_2+=1
            test_window_2.pop(0)
            test_window_2.append(features[idx][1])
        else:
            test_window_2.append(features[idx][1])
        #feature 3
        if len(test_window_3)==w_size:
            res = cdd_3(test_window_3, reference_window_3)

            if res[1]<confidence:
                if idx>=drift_index:
                    tp_3 = idx
                    return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3
                else:
                    fp_count_3+=1
            test_window_3.pop(0)
            test_window_3.append(features[idx][2])
        else:
            test_window_3.append(features[idx][2])

    return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3

def river_cd(features, drift_index, cdd_1, cdd_2, cdd_3):
    fp_count_1, fp_count_2, fp_count_3 = 0,0,0
    tp_1, tp_2, tp_3 = None, None, None

    for idx in range(len(features[1500:])):
        idx = idx+1500

        #first feature
        cdd_1.update(features[idx][0])
        if cdd_1.drift_detected:
            if idx >= drift_index:
                tp_1 = idx

                return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3
            
            else:
                fp_count_1+=1

        #second feature
        cdd_2.update(features[idx][1])
        if cdd_2.drift_detected:
            if idx >= drift_index:
                tp_2 = idx

                return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3
            
            else:
                fp_count_2+=1

        #third feature
        cdd_3.update(features[idx][2])
        if cdd_3.drift_detected:
            if idx >= drift_index:
                tp_3 = idx

                return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3
            
            else:
                fp_count_3+=1

    

    return tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count_3



def error_rate_drift(dataset_name, drift_index):

    #retrieve actual labels and predicted labels
    ddm_tp, ddm_fp, ddm_missed =[], [], []
    eddm_tp, eddm_fp, eddm_missed =[], [], []
    adwin_tp, adwin_fp, adwin_missed =[], [], []
    kswin_tp, kswin_fp, kswin_missed =[], [], []
    mannu_tp, mannu_fp, mannu_missed =[], [], []
    ks_tp, ks_fp, ks_missed =[], [], []


    for i in range(1,6):

        print(i, "next dataset")
        features = np.load(f"datasets/features_{dataset_name}_{i}.npy")

        #ddm
        ddm_drift_1 = drift.binary.DDM()
        ddm_drift_2 = drift.binary.DDM()
        ddm_drift_3 = drift.binary.DDM()
        tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3 = river_cd(features, drift_index, ddm_drift_1,ddm_drift_2, ddm_drift_3)
        print(tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3)

        #eddm
        eddm_drift_1 = drift.binary.EDDM()
        eddm_drift_2 = drift.binary.EDDM()
        eddm_drift_3 = drift.binary.EDDM()
        tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3 = river_cd(features, drift_index, eddm_drift_1, eddm_drift_2, eddm_drift_3)
        print(tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3)

        #adwin
        adwin_drift_1 = drift.ADWIN()
        adwin_drift_2 = drift.ADWIN()
        adwin_drift_3 = drift.ADWIN()
        tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3 = river_cd(features, drift_index, adwin_drift_1, adwin_drift_2, adwin_drift_3)
        print(tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3)
        
        #kswin
        kswin_drift_1 = drift.KSWIN(window_size=200, stat_size=100)
        kswin_drift_2 = drift.KSWIN(window_size=200, stat_size=100)
        kswin_drift_3 = drift.KSWIN(window_size=200, stat_size=100)
        tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3 = river_cd(features, drift_index, kswin_drift_1, kswin_drift_2, kswin_drift_3)
        print(tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3)      

        #mannu
        w_size = 100
        confidence = 0.005
        mannu_1 = stats.mannwhitneyu
        mannu_2 = stats.mannwhitneyu
        mannu_3 = stats.mannwhitneyu
        tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3 = custom_cd(features, drift_index, w_size, confidence, mannu_1, mannu_2, mannu_3)
        print(tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3) 

        #ks
        w_size = 100
        confidence = 0.005
        ks_1 = stats.ks_2samp
        ks_2 = stats.ks_2samp
        ks_3 = stats.ks_2samp
        tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3 = custom_cd(features, drift_index, w_size, confidence, ks_1, ks_2, ks_3)
        print(tp_1, tp_2, tp_3, fp_count_1, fp_count_2, fp_count3) 

def main():

    #params

    dataset_name = "SEA_1_2"
    drift_index = 5000
    
    #3 functions for error-rate, features and 3-drift

    error_rate_drift(dataset_name, drift_index)
    #data = [20,26]
    #print(np.mean(data), np.std(data))
if __name__ == '__main__':
    main()
