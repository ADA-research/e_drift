import pandas as pd
import numpy as np
from river import drift
from scipy import stats


def retrieve_data(dataset_name):

    labels = np.load(f"datasets/labels_SEA_{dataset_name}.npy")
    epsilon_data = pd.read_csv(f"results/SEA_{dataset_name}_correct.csv")
    epsilons = epsilon_data["epsilon"].to_list()
    print(len(epsilons[900:]))
    return labels, epsilons[900:]

def update_det2(epsilon, idx, reference, detection, window_size, fp_count):

    #first fill reference window
    tp = None

    if len(reference) != window_size:
        reference.append(epsilon)
    
    else:

        if len(detection)==window_size:
            res = stats.mannwhitneyu(reference, detection)
            if res[1]< 0.005:

                if idx>=5000:
                    tp = idx
                    return reference, detection, tp, fp_count
                else:
                    fp_count+=1
            
            detection.pop(0)
            detection.append(epsilon)
        
        else:
            detection.append(epsilon)

    return reference, detection, tp, fp_count


def detect_drift2(labels, epsilons, drift_index, window_size):

    reference_true, reference_false = [], []
    detection_true, detection_false = [], []
    fp_count = 0
    tp = None
    class_label = None

    for idx, (label, epsilon) in enumerate(zip(labels[1900:6000], epsilons)):
        idx+=1900

        #split based on class
        if label==True:
            reference_true, detection_true, tp, fp_count = update_det2(epsilon, idx, reference_true, detection_true, window_size, fp_count)
            if tp != None:
                class_label = True
            
            
        else:
            reference_false, detection_false, tp, fp_count = update_det2(epsilon, idx, reference_false, detection_false, window_size, fp_count)
            if tp != None:
                class_label = False
        
        if tp != None:
            print("drift detection index: ", tp)
            print("FPS: ", fp_count)
            print("Detected class: ", class_label)
            break
    
    print("No drift detected within 1000 instances")
    print("FPS: ", fp_count)

def update_det(epsilon, idx, reference, detection, window_size, fp_count):

    #first fill reference window
    tp = None

    if len(reference) != window_size:
        reference.append(epsilon)
    
    else:

        if len(detection)==window_size:
            res = stats.ks_2samp(reference, detection)
            if res[1]< 0.005:

                if idx>=5000:
                    tp = idx
                    return reference, detection, tp, fp_count
                else:
                    fp_count+=1
            
            detection.pop(0)
            detection.append(epsilon)
        
        else:
            detection.append(epsilon)

    return reference, detection, tp, fp_count


def detect_drift(labels, epsilons, drift_index, window_size):

    reference_true, reference_false = [], []
    detection_true, detection_false = [], []
    fp_count = 0
    tp = None
    class_label = None

    for idx, (label, epsilon) in enumerate(zip(labels[1900:6000], epsilons)):
        idx+=1900

        #split based on class
        if label==True:
            reference_true, detection_true, tp, fp_count = update_det(epsilon, idx, reference_true, detection_true, window_size, fp_count)
            if tp != None:
                class_label = True
            
            
        else:
            reference_false, detection_false, tp, fp_count = update_det(epsilon, idx, reference_false, detection_false, window_size, fp_count)
            if tp != None:
                class_label = False
        
        
        
        
        if tp != None:
            print("drift detection index: ", tp)
            print("FPS: ", fp_count)
            print("Detected class: ", class_label)
            break
    
    print("No drift detected within 1000 instances")
    print("FPS: ", fp_count)

        
def detect_drift_3(labels, epsilons, drift_index, window_size):

    fp_count = 0
    tp = None
    class_label = None

    ddm_true = drift.binary.DDM()
    ddm_false = drift.binary.DDM()

    for idx, (label, epsilon) in enumerate(zip(labels[1900:6000], epsilons)):
        idx+=1900

        if label == True:
            ddm_true.update(epsilon)
            if ddm_true.drift_detected:

                if idx >=drift_index:
                    tp = idx
                    class_label = True
                    print(tp, fp_count, class_label)
                else:
                    fp_count+=1

        else:
            ddm_false.update(epsilon)
            if ddm_false.drift_detected:
                if idx>=drift_index:
                    tp = idx
                    class_label = False
                    print(tp, fp_count, class_label)
                else:
                    fp_count+=1

    print(tp, fp_count, class_label, "no drift")



def main():

    #params

    dataset_name = "2_3_5"
    drift_index = 5000
    window_size = 100

    labels, epsilons = retrieve_data(dataset_name)
    # detect_drift(labels, epsilons, drift_index, window_size)
    # detect_drift2(labels, epsilons, drift_index, window_size)
    # detect_drift_3(labels, epsilons, drift_index, window_size)

    mtd = [40,53,109,101,81]
    fac = [0,2,0,0,50]

    print(np.mean(mtd), np.std(mtd))
    print(np.mean(fac), np.std(fac))
    
    

if __name__ == '__main__':
    main()