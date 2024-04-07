import yaml
import numpy as np



#other python files

#TRAIN NEURAL NETWORK

#VALIDATE NEURAL NETWORK

#GENERATE PREDICTION LABELS

#CONCEPT DRIFT DETECTION
def train_nn():
    pass

def create_dataset(dataset_params: dict, dataset_name: str):
    """Generate dataset with drift."""

    if dataset_name == "SEA":
        from data_generator import SEA
        dataset_params = dataset_params[dataset_name]
        dataset = SEA(dataset_params["variant_1"], dataset_params["variant_2"], 
                      dataset_params["noise"], dataset_params["seed"], 
                      dataset_params["normalization"])
    
    elif dataset_name == "HYP":
        from data_generator import HYP
        dataset_params = dataset_params[dataset_name]
        dataset = HYP(dataset_params["n_features"], dataset_params["n_drift_features"], 
                      dataset_params["mag_change"], dataset_params["sigma"], 
                      dataset_params["noise"], dataset_params["seed"],
                      dataset_params["normalization"])

    else:
        print(f"{dataset_name} is not (yet) a supported dataset.")
    
    dataset.generate_dataset(dataset.variant_1, dataset.variant_2, dataset_params["name"])

def read_main_config(filename: str) -> yaml:
    """Read in the main configuration file."""
    with open(f"{filename}.yaml", "r") as f:
        main_config = yaml.safe_load(f)
    return main_config

#MAIN
def main():
    filename = "main_config"

    #read in the main_config file
    main_config = read_main_config(filename)

    #retrieve dataset name

    #build in option to skip dataset generator with simple if statement
    dataset_name = list(main_config["general"]["data_generator"].keys())[0]

    #generate dataset
    create_dataset(main_config["general"]["data_generator"], dataset_name)

    #train neural network



if __name__ == '__main__':
    main()
