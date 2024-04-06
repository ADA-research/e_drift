import yaml




#GENERATE DATA

#TRAIN NEURAL NETWORK

#VALIDATE NEURAL NETWORK

#CONCEPT DRIFT DETECTION



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
    print(main_config)



if __name__ == '__main__':
    main()
