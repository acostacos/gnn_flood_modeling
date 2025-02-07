# TODO: REMOVE THIS
import warnings
warnings.filterwarnings('ignore')
# --------------------------------

import yaml
import traceback

from argparse import ArgumentParser
from data.temporal_graph_dataset import TemporalGraphDataset

def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config file')
    return parser

def main():
    try:
        parser = get_arg_parser()
        args = parser.parse_args()

        with open(args.config_path) as f:
            config = yaml.safe_load(f)
        
        dataset = TemporalGraphDataset(node_features=config['node_features'],
                            edge_features=config['edge_features'],
                            **config['dataset_parameters']).load()
        print(len(dataset))
    except yaml.YAMLError as e:
        print('Error loading config YAML file. Error: ', e)
    except Exception:
        print('Unexpected error:\n', traceback.format_exc())


if __name__ == '__main__':
    main()
