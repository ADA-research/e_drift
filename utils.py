import yaml
import torch.optim as optim

def write_yaml_HPO(yaml_name, params):

    #open yaml file
    with open(yaml_name, "r") as file:
        main_config = yaml.safe_load(file)

    #include best found hyperparameters
    main_config["general"]["training"]["batch_size"] = params["batch_size"]
    main_config["general"]["training"]["epochs"] = params["max_epochs"]
    main_config["general"]["training"]["learning_rate"] = params["optimizer__lr"]

    #special case for optimizer
    if params["optimizer"] == optim.SGD:
        main_config["general"]["training"]["optimizer"] = "SGD"

    elif params["optimizer"] == optim.RMSprop:
        main_config["general"]["training"]["optimizer"] = "RMSprop"


    elif params["optimizer"]== optim.Adam:
        main_config["general"]["training"]["optimizer"] = "Adam"
    
    else:
        print("does not exist")
    

    #write to the yaml file after adjustements
    with open(yaml_name, "w") as outfile:
        yaml.dump(main_config, outfile, default_flow_style=False, sort_keys = False)
    

