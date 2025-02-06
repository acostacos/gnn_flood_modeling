# TODO: REMOVE THIS
import warnings
warnings.filterwarnings('ignore')
# --------------------------------

import yaml

from argparse import ArgumentParser
from data.flood_dataset import FloodDataset

def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config file')
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    with open(args.config_path) as f:
        try:
            config = yaml.safe_load(f)
        except:
            print('Error loading config YAML file.')
    
    dataset = FloodDataset(node_features=config['node_features'],
                           edge_features=config['edge_features'],
                           **config['dataset_parameters']).load()
    print(len(dataset))


if __name__ == '__main__':
    main()
