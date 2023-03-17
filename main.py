import yaml
import argparse
import pandas as pd
from src.get_data import DataFetcher 

def read_params(config_path):
    """Read parameters from the parameters.yaml file
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
#
#
#
#   
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i","--input", default="parameters.yaml")
    parsed_args = args.parse_args()
    param_dict = read_params(config_path=parsed_args.input)
    DataFetcher(param_dict['data']['train']).read_file()
    

