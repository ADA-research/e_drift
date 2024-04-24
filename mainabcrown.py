import os
import subprocess
import yaml
import pickle
import time
import csv
import numpy as np
from pathlib import Path

def generate_yaml(begin_idx, end_idx, epsilon, timeout, yaml_name):
    #open the current yaml file
    with open(yaml_name, "r") as file:
        abcrown_config = yaml.safe_load(file)
    
    #set maximum number of seconds to perform search
    abcrown_config["bab"]["timeout"]= timeout

    #adjust instance if necessary
    abcrown_config["data"]["start"] = begin_idx
    abcrown_config["data"]["end"] = end_idx

    #adjust epsilon if necessary
    abcrown_config["specification"]["epsilon"] = epsilon

    #write to the yaml file after adjustements
    with open(yaml_name, "w") as outfile:
        yaml.dump(abcrown_config, outfile, default_flow_style=False, sort_keys = False)

def run_abcrown(yaml_name):
    command = f"python abcrown.py --config {yaml_name}" #srun -p graceTST python abcrown.py --config exp_configs/tutorial_examples/custom_nn.yaml
    result = subprocess.run([command], shell=True, capture_output=True, text=True)
    #print(result.stderr)
    #print(result.stdout)

def read_results(result_file):
    # reading the data from the saved file 
    with open(result_file, 'rb') as f: 
        data = pickle.load(f)
    return data["results"]

def determine_next_eps(eps_idx, epsilons, safe_idx, unsafe_idx):
    
    # if both are still np inf and -np inf, pick eps in the middle

    if safe_idx == -np.inf and unsafe_idx == np.inf:
        print("allebei -np inf and np inf")
        eps_idx = np.random.choice(epsilons)

    #check if safe_idx is still -np inf
    elif safe_idx == -np.inf:
        print("safe idx is - np.inf")
        eps_idx = int(eps_idx/2)

    #check if unsafe_idx is still np.inf
    elif unsafe_idx == np.inf:
        print("unsafe idx is np inf")
        eps_idx = int((len(epsilons)+eps_idx)/2)

    #otherwise both safe_idx and unsafe_idx are set so pick epsilon in between
    else:
        print("safe and unsafe zijn gezet")
        eps_idx = int((safe_idx + unsafe_idx)/2)
    
    return eps_idx

def unknown_search(instance_idx, eps_idx, epsilons, safe_idx, unsafe_idx, timeout, yaml_name, result_file):

    if safe_idx == -np.inf:
        begin = 0
    else:
        #safe_idx +1 since we do not need to evaluate it again
        begin = safe_idx+1
    if unsafe_idx == np.inf:
        end = len(epsilons)
    else:
        #unsafe_idx not -1 since in range it is until end (so actually end-1)
        end = unsafe_idx
    
    
    for eps_idx in range(begin, end):

        epsilon = epsilons[eps_idx]

        generate_yaml(instance_idx, instance_idx+1, epsilon, timeout, yaml_name)
        #perform ABCROWN
        run_abcrown(yaml_name)

        #read the results
        result = read_results(result_file)
        print("resultaat: ", result)

        #determine if result is safe, unsafe, error or out of time

        if result[0][0] == "safe-incomplete" or result[0][0] == "safe":
            #the result is verified or safe/unsat
            if eps_idx > safe_idx:
                safe_idx = eps_idx

        elif result[0][0] == "unsafe-pgd"  or result[0][0] == "unsafe":
            #the result is falsified or unsafe/sat
            if eps_idx < unsafe_idx:
                unsafe_idx = eps_idx

        else:
            print(result[0], "dit is een nieuwe error")
            print(safe_idx, unsafe_idx)
            print(eps_idx, epsilons[eps_idx])
            #break

    return safe_idx, unsafe_idx 


def determine_critical_eps(instance_idx, epsilons, timeout, yaml_name, result_file):

    #always start with middle epsilon value
    eps_idx = int(len(epsilons)/2)
    epsilon = epsilons[eps_idx]

    safe_idx = -np.inf
    unsafe_idx = np.inf

    terminated = False
    counter = 0

    while not terminated:
        print(counter, "counter")
        print("epsilon: ", epsilon)

        #check terminate conditions

        # if safe_idx == unsafe_idx -1 -> we found two next to each other
        if safe_idx == unsafe_idx -1:
            print("terminated: next to each other")
            return safe_idx, unsafe_idx

        # if safe_idx == len(epsilons) - 1 -> all epsilons are safe
        elif safe_idx == len(epsilons)-1:
            print("terminated: everything is safe")
            return safe_idx, unsafe_idx

        # if unsafe_idx == 0 -> all epsilons are unsafe
        elif unsafe_idx == 0:
            print("temrnated: everythig is unsafe")
            return safe_idx, unsafe_idx

        #generate the correct yaml file
        generate_yaml(instance_idx, instance_idx+1, epsilon, timeout, yaml_name)

        #perform ABCROWN
        run_abcrown(yaml_name)

        #read the results
        result = read_results(result_file)
        print("resultaat: ", result)

        #determine if result is safe, unsafe, error or out of time

        if result[0][0] == "safe-incomplete" or result[0][0] == "safe":
            #the result is verified or safe/unsat
            if eps_idx > safe_idx:
                safe_idx = eps_idx

        elif result[0][0] == "unsafe-pgd"  or result[0][0] == "unsafe":
            #the result is falsified or unsafe/sat
            if eps_idx < unsafe_idx:
                unsafe_idx = eps_idx

        elif result[0][0] == "unknown":
            return unknown_search(instance_idx, eps_idx, epsilons, safe_idx, unsafe_idx, timeout, yaml_name, result_file)

        else:
            print(result[0], "dit is een nieuwe error")
            print(safe_idx, unsafe_idx)
            print(eps_idx, epsilons[eps_idx])
            #break
        
        #determine new eps idx
        eps_idx = determine_next_eps(eps_idx, epsilons, safe_idx, unsafe_idx) 
        epsilon = epsilons[eps_idx]
        
        counter+=1

        if counter == len(epsilons):
            terminated = True
    
    return safe_idx, unsafe_idx

def main():

    #set begin and end indexis
    begin_idx = 0
    end_idx = 10000

    #epsilon values according to Bosman et al. 
    epsilons = np.around(np.arange(0.001, 0.4, 0.002), decimals=3)
    epsilons = epsilons.tolist()
    
    #define timeout in seconds
    timeout = 600

    #load dataset and predicted labels
    dataset_name = "SEA_0_1"
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom')

    features = np.load(database_path + f"/datasets/features_{dataset_name}.npy")
    labels = np.load(database_path + f"/datasets/labels_{dataset_name}.npy")
    labels_pred = np.load(database_path + f"/datasets/labels_{dataset_name}_pred.npy")

    #yaml file name
    yaml_name = f"exp_configs/tutorial_examples/custom_{dataset_name}.yaml"
    result_file = f'custom_{dataset_name}.txt'

    #generate csv file
    csv_file = Path(f"results/{dataset_name}.csv")
    write_header = not csv_file.exists()
    with open(csv_file, "a") as file:
        writer = csv.writer(file, ["instance_idx","instance", "epsilon", "runtime"])
        if write_header:
            writer.writerow(["instance_idx","instance", "epsilon", "runtime"])
    
    #compute epsilon distances
    for idx in range(begin_idx, end_idx):
        print("index: ", idx)
        
        #check if ground truth and predicted label correspond
        if labels[idx] != labels_pred[idx]:

            #get distance towards decision boundary for misclassified instance
            query_time = time.time()
            safe_idx, unsafe_idx = determine_critical_eps(idx, epsilons, timeout, yaml_name, result_file)
            print(safe_idx, unsafe_idx, "end")
            query_time = time.time() - query_time

            #check if safe is not set:
            if safe_idx == -np.inf:
                #check if instance is very close to decision boundary (safe==-np.inf and unsafe==0)
                if unsafe==0:
                    safe_eps=0
            
                #otherwise i) (unknown....unsafe) or ii) all are unknown
                else:
                    #actually we want to skip this instance
                    safe_eps=-1
            else:
                safe_eps = epsilons[safe_idx]
                
            with open(csv_file, "a") as file:
                writer = csv.writer(file, ["instance_idx","instance", "epsilon", "runtime"])
                writer.writerow([idx, features[idx], safe_eps, query_time])

        else:
            #correctly classified instances are automatically set to 0
            with open(csv_file, "a") as file:
                writer = csv.writer(file, ["instance_idx","instance", "epsilon", "runtime"])
                writer.writerow([idx, features[idx], 0, 0])

    print("experiment finished")

if __name__ == '__main__':


    main()