import traceback
import yaml

from argparse import ArgumentParser, Namespace
from data import FloodingEventDataset
from utils import Logger

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='configs/preprocess_config.yaml', help='Path to preprocessing config file')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = Logger(log_path=args.log_path)

    try:
        with open(args.config_path) as f:
            config = yaml.safe_load(f)

        output_parameters = config['output_parameters']
        if 'output_file_path' not in output_parameters or output_parameters['output_file_path'] is None:
            raise ValueError('output_file_path not specified in config file.')

        flood_events = config['flood_event_parameters']['hec_ras_hdf_paths']
        for key, hec_ras_path in flood_events.items():
            dataset = FloodingEventDataset(hec_ras_hdf_path=hec_ras_path,
                                        node_features=config['node_features'],
                                        edge_features=config['edge_features'],
                                        debug=args.debug,
                                        logger=logger,
                                        **config['mesh_parameters'],
                                        **config['dataset_parameters'])
            dataset.load()
            dataset.save(output_parameters['output_file_path'], key)
    except yaml.YAMLError as e:
        logger.log(f'Error loading config YAML file. Error: {e}')
    except ValueError as e:
        logger.log(f'ValueError: {e}')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
