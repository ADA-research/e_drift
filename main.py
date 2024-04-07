import yaml
import numpy as np

#other python files
from neural_networks import Train

#CONCEPT DRIFT DETECTION

def train_nn(network_params: dict, dataset_name: str):
    
    instance = Train(dataset_name, network_params["training_instances"],
                     network_params["shuffle"], network_params["model"],
                     network_params["batch_size"], network_params["epochs"],
                     network_params["learning_rate"], network_params["visualize"])
    
    instance.train_pipeline()

def create_dataset(dataset_params: dict, synth_dataset: str, dataset_name: str):
    """Generate dataset with drift."""

    if synth_dataset == "SEA":
        from data_generator import SEA
        dataset_params = dataset_params[synth_dataset]
        dataset = SEA(dataset_params["variant_1"], dataset_params["variant_2"], 
                      dataset_params["noise"], dataset_params["seed"], 
                      dataset_params["normalization"])
    
    elif synth_dataset == "HYP":
        from data_generator import HYP
        dataset_params = dataset_params[synth_dataset]
        dataset = HYP(dataset_params["n_features"], dataset_params["n_drift_features"], 
                      dataset_params["mag_change"], dataset_params["sigma"], 
                      dataset_params["noise"], dataset_params["seed"],
                      dataset_params["normalization"])

    else:
        print(f"{synth_dataset} is not (yet) a supported dataset.")
    
    dataset.generate_dataset(dataset.var_1, dataset.var_2, dataset_name)

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
    synth_dataset = list(main_config["general"]["data_generator"].keys())[0]
    dataset_name = main_config["general"]["dataset_name"]

    #generate dataset
    create_dataset(main_config["general"]["data_generator"], synth_dataset, dataset_name)

    #train neural network and get predications based on trained neural network
    train_nn(main_config["general"]["training"], dataset_name)


if __name__ == '__main__':
    main()
