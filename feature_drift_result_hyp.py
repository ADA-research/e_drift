import pandas as pd
import numpy as np
from river import drift
from scipy import stats


def custom_cd(features, drift_index, mannu, w_size, confidence):

    fp_counts=  [0 for i in range(len(features[0]))]
    tps = [None for i in range(len(features[0]))]


    reference_windows = [[] for i in range(10)]
    test_windows = [[] for i in range(10)]

    for idx in range(len(features[1900:2000])):

        idx+=1900

        for f_idx in range(10):

            reference_windows[f_idx].append(features[idx][f_idx])
    

    for idx in range(len(features[2000:6000])):

        idx=idx+2000

        for f_idx, feature in enumerate(features[idx]):


            if len(test_windows[f_idx])==w_size:
                res = mannu(test_windows[f_idx], reference_windows[f_idx])

                if res[1]<confidence:
                    if idx>=drift_index:
                        tps[f_idx] = idx
                        return fp_counts, tps
                    else:
                        fp_counts[f_idx]+=1
                
                test_windows[f_idx].pop(0)
                test_windows[f_idx].append(feature)
            else:
                test_windows[f_idx].append(feature)
    return fp_counts, tps

def river_cd(features, drift_index, cd_detectors):

    fp_counts=  [0 for i in range(len(features[0]))]
    tps = [None for i in range(len(features[0]))]

    for idx in range(len(features[1900:6000])):
        idx = idx+1900

        for f_idx, feature in enumerate(features[idx]):

            cdd = cd_detectors[f_idx]
            cdd.update(feature)

            if cdd.drift_detected:

                if idx >= drift_index:
                    tps[f_idx] = idx

                    return fp_counts, tps
                
                else:
                    fp_counts[f_idx] +=1
    return fp_counts, tps





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


        ddm_detectors = [drift.binary.DDM() for i in range(10)]

        fp_counts, tps = river_cd(features, drift_index, ddm_detectors)
        print("DDM")
        print(fp_counts)
        print(tps)

    for i in range(1,6):
        features = np.load(f"datasets/features_{dataset_name}_{i}.npy")
        adwin_detectors = [drift.ADWIN() for i in range(10)]

        fp_counts, tps = river_cd(features, drift_index, adwin_detectors)
        print("ADWIN")
        print(fp_counts)
        print(tps)

        
    for i in range(1,6):
        features = np.load(f"datasets/features_{dataset_name}_{i}.npy")
        kswin_detectors = [drift.KSWIN(window_size=200, stat_size=100) for i in range(10)]

        fp_counts, tps = river_cd(features, drift_index, kswin_detectors)
        print("KSWIN")
        print(fp_counts)
        print(tps)


    for i in range(1,6):
        features = np.load(f"datasets/features_{dataset_name}_{i}.npy")
        mannu = stats.ks_2samp
        w_size=100
        confidence = 0.005

        fp_counts, tps = custom_cd(features, drift_index, mannu, w_size, confidence)
        print("MANNU")
        print(fp_counts)
        print(tps)





def main():

    #params

    dataset_name = "HYP_001"
    drift_index = 5000
    
    #3 functions for error-rate, features and 3-drift

    error_rate_drift(dataset_name, drift_index)
    #data = [135,154,29,228,3]
    #print(np.mean(data), np.std(data))
if __name__ == '__main__':
    main()