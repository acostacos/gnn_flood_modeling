import numpy as np
import traceback
import torch
import yaml

from argparse import ArgumentParser, Namespace
from data import TemporalGraphDataset
from models import SWEGNN

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()

def main():
    try:
        args = parse_args()

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

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
