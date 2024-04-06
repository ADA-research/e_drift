import yaml
import numpy as np



#other python files
from data_generator import SEA



#GENERATE DATA

#TRAIN NEURAL NETWORK

#VALIDATE NEURAL NETWORK

#CONCEPT DRIFT DETECTION

def generate_dataset(dataset_name: str) -> np.ndarray:

    if dataset_name == "SEA":
        pass
    
    elif dataset_name == "HYP":
        pass

    else:
        print(f"{dataset_name} is not a supported dataset.")


    pass

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
    dataset_name = list(main_config["general"]["data_generator"].keys())[0]

    #generate dataset
    generate_dataset(dataset_name)


if __name__ == '__main__':
    main()
