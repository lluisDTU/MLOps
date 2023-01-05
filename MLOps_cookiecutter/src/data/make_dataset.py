# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import torch
import numpy as np

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load the raw data files
    raw_data_x = []
    raw_data_y = []

    # Get the list of files in the directory
    filenames = os.listdir(input_filepath)

    # Remove the .DS_Store file from the list
    filenames = [f for f in filenames if f != '.DS_Store' and f != '.gitkeep']

    for file in filenames:
        # Load the file as a numpy array
        data = np.load(os.path.join(input_filepath, file))
        raw_data_x.append(data['images'])
        raw_data_y.append(data['labels'])

    # Concatenate the data into a single tensor
    tensor = torch.tensor(np.array(raw_data_x))

    # Normalize the tensor (make the mean 0 and standard deviation 1)
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std

    # Save the normalized tensor to the data/processed folder
    torch.save(normalized_tensor, os.path.join(output_filepath, 'normalized_tensor.pt'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # Get the root directory of the repository
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Construct the path to the raw directory
    raw_dir = os.path.join(repo_root, 'data', 'raw')
    processed_dir = os.path.join(repo_root, 'data', 'processed')
    main(raw_dir,processed_dir)
