import pandas as pd
import numpy as np
from river import drift
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def e_custom_cd(epsilons, drift_index, w_size, confidence, cdd):

    reference_window = epsilons[1900:2000]
    test_window = []
    fp_count = 0
    tp = None

    for idx, epsilon in enumerate(epsilons[2000:]):

        idx = idx+2000


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


def e_river_cd(epsilons, drift_index, cdd):
    fp_count = 0
    tp = None

    for idx, epsilon in enumerate(epsilons[1900:]):
        idx =idx+1900

        cdd.update(epsilon)

        if cdd.drift_detected:

            if idx >= drift_index:
                tp = idx

                return tp, fp_count
            
            else:
                fp_count+=1
    return tp, fp_count

def error_custom_cd(labels, labels_pred, drift_index, w_size, confidence, cdd):

    reference_window, test_window = [], []
    fp_count = 0
    tp = None

    for idx in range(len(labels[1900:2000])):
        idx = idx+1900

        reference_window.append(int(not labels[idx] == labels_pred[idx]))
   

    for idx in range(len(labels[2000:])):

        idx = idx+2000

        result = int(not labels[idx] == labels_pred[idx])


        if len(test_window)==w_size:

            res = cdd(test_window, reference_window)

            if res[1]< confidence:

                if idx>=drift_index:
                    tp = idx
                    return tp, fp_count
                else:
                    fp_count+=1
            
            test_window.pop(0)
            test_window.append(result)
        
        else:
            test_window.append(result)
    return tp, fp_count

def error_river_cd(labels, labels_pred, drift_index, cdd):
    fp_count = 0
    tp = None

    for idx in range(len(labels[1900:])):
        idx = idx+1900

        result = int(not labels[idx] == labels_pred[idx])
        cdd.update(result)

        if cdd.drift_detected:

            if idx >= drift_index:
                tp = idx

                return tp, fp_count
            
            else:
                fp_count+=1
    return tp, fp_count

def results_custom(dataset_name, drift_index, w_size, confidence, drift_det):
    
    error_tp, error_fp = [], []
    e_tp, e_fp = [], []
    for i in range(1,6):
        #retrieve labels for error rate
        labels = np.load(f"datasets/labels_{dataset_name}_{i}.npy")
        labels_pred = np.load(f"datasets/labels_{dataset_name}_{i}_pred.npy")

        #retrieve epsilon values
        data = pd.read_csv(f"results/{dataset_name}_{i}.csv")
        epsilons = data["epsilon"].to_list()

        tp, fp = error_custom_cd(labels, labels_pred, drift_index, w_size, confidence, drift_det)
        if tp != None:
            error_tp.append(tp)
        error_fp.append(fp)

        tp, fp = e_custom_cd(epsilons, drift_index, w_size, confidence, drift_det)
        if tp != None:
            e_tp.append(tp)
        e_fp.append(fp)

    return np.mean(error_tp)-drift_index, np.mean(error_fp), np.mean(e_tp)-drift_index, np.mean(e_fp)

def results_river(dataset_name, drift_index, error_drift_dets, e_drift_dets):

    error_tp, error_fp = [], []
    e_tp, e_fp = [], []

    for i in range(0,5):
        #retrieve labels for error rate
        labels = np.load(f"datasets/labels_{dataset_name}_{i+1}.npy")
        labels_pred = np.load(f"datasets/labels_{dataset_name}_{i+1}_pred.npy")

        #retrieve epsilon values
        data = pd.read_csv(f"results/{dataset_name}_{i+1}.csv")
        epsilons = data["epsilon"].to_list()


        error_drift = error_drift_dets[i]
        tp, fp = error_river_cd(labels, labels_pred, drift_index, error_drift)
        if tp != None:
            error_tp.append(tp)
        error_fp.append(fp)

        e_drift = e_drift_dets[i]
        tp, fp = e_river_cd(epsilons, drift_index, e_drift)
        if tp != None:
            e_tp.append(tp)
        e_fp.append(fp)
    


    return np.mean(error_tp)-drift_index, np.mean(error_fp), np.mean(e_tp)-drift_index, np.mean(e_fp)


def graph1(tp_positions, fp_counts, confidences):
    plt.rcParams["figure.figsize"] = (5,5)
    names = ["error-rate + DDM", "e-drift + DDM", "error-rate + ADWIN", "e-drift + ADWIN", "error-rate + MannU", "e-drift + MannU", "error-rate + KS", "e-drift + KS"]
    fig1, ax1 = plt.subplots()
    for i in range(len(names)):
        new_tp, new_fp = [], []
        # for tp, fp in zip(tp_positions[i], fp_counts[i]):
        #     if tp <1500:
        #         new_tp.append(tp)
        #         new_fp.append(fp)
        # tp_positions[i] = new_tp
        # fp_counts[i] = new_fp
        x_points = np.empty(len(tp_positions[i]))
        x_points.fill(i)
        color_list = []
        for fp in fp_counts[i]:

            if fp==0:
                color_list.append("green")

            elif fp <5:
                color_list.append("yellow")
            
            elif fp <10:
                color_list.append("orange")
            
            else:
                color_list.append("red")

        ax1.scatter(x_points, tp_positions[i], c=color_list, cmap="viridis")
        ax1.plot(x_points, tp_positions[i], linestyle = "-", color="black")


        for j in range(len(tp_positions[i])):
            #ax1.plot(x_points[j], tp_positions[i][j], marker = "o", c=color_list[j])
            ax1.annotate(confidences[j], (x_points[j],tp_positions[i][j]), ha='left', rotation=60)
    green_patch = mpatches.Patch(color='green', label='FPS = 0')
    yellow_patch = mpatches.Patch(color='yellow', label='FPS < 5')
    orange_patch = mpatches.Patch(color='orange', label='FPS < 10')
    red_patch = mpatches.Patch(color='red', label='FPS >= 10')
    x_ticks = np.arange(0,8)
    plt.xticks(x_ticks, names)
    plt.xlabel("detection mechanism")
    plt.ylabel("TP detection position")
    #plt.ylim(-0.1, 1500)
    plt.title("HYP m=0.001")
    plt.yscale("log")
    plt.margins(0.2)
    plt.legend(handles=[green_patch, yellow_patch, orange_patch, red_patch])
    #plt.colorbar()
    plt.show()




def main():

    #params

    dataset_name = "SEA_1_2"
    drift_index = 5000

    #save results
    tp_positions, fp_counts = [], []
    
    #DDM + EDDM
    error_drift_dets =[drift.binary.DDM() for i in range(5)]
    e_drift_dets = [drift.binary.DDM() for i in range(5)]
    drift_det = drift.binary.DDM

    error_tp, error_fp, e_tp, e_fp = results_river(dataset_name, drift_index, error_drift_dets, e_drift_dets)
    tp_positions.append([error_tp])
    tp_positions.append([e_tp])
    fp_counts.append([error_fp])
    fp_counts.append([e_fp])

    #ADWIN
    confidences = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    error_tps, error_fps = [], []
    e_tps, e_fps = [], []
    for confidence in confidences:
        error_drift_dets = [drift.ADWIN(delta=confidence) for i in range(5)]
        e_drift_dets = [drift.ADWIN(delta=confidence) for i in range(5)]
        error_tp, error_fp, e_tp, e_fp = results_river(dataset_name, drift_index, error_drift_dets, e_drift_dets)
        error_tps.append(error_tp)
        e_tps.append(e_tp)
        error_fps.append(error_fp)
        e_fps.append(e_fp)
    tp_positions.append(error_tps)
    tp_positions.append(e_tps)
    fp_counts.append(error_fps)
    fp_counts.append(e_fps)

    #MANNU + KS
    confidences = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    error_tps, error_fps = [], []
    e_tps, e_fps = [], []
    drift_det = stats.mannwhitneyu
    w_size = 100
    for confidence in confidences:
        print(confidence)
        error_tp, error_fp, e_tp, e_fp = results_custom(dataset_name, drift_index, w_size, confidence, drift_det)
        error_tps.append(error_tp)
        e_tps.append(e_tp)
        error_fps.append(error_fp)
        e_fps.append(e_fp)
    tp_positions.append(error_tps)
    tp_positions.append(e_tps)
    fp_counts.append(error_fps)
    fp_counts.append(e_fps)

    confidences = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    error_tps, error_fps = [], []
    e_tps, e_fps = [], []
    drift_det = stats.ks_2samp
    w_size = 100
    for confidence in confidences:
        print(confidence)
        error_tp, error_fp, e_tp, e_fp = results_custom(dataset_name, drift_index, w_size, confidence, drift_det)
        error_tps.append(error_tp)
        e_tps.append(e_tp)
        error_fps.append(error_fp)
        e_fps.append(e_fp)
    tp_positions.append(error_tps)
    tp_positions.append(e_tps)
    fp_counts.append(error_fps)
    fp_counts.append(e_fps)


    graph1(tp_positions, fp_counts, confidences)


if __name__ == '__main__':
    main()