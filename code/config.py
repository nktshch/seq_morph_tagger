"""
config.py contains dictionary with all necessary hyperparameters as well as paths to necessary files
"""

import argparse


config = {}

config['embeddings_dimension'] = 300 # fastText embeddings
config['sentence_batch_size'] = 5
# add all other paths (for pickle files)

def configurate():
    """
    Adds command line arguments to config dictionary.
    """
    parse_arguments()

def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument('train_files', nargs='+')
    argp.add_argument('--sentences_files', nargs='+')

    args = argp.parse_args()
    config['train_files'] = args.train_files
    config['sentences_files'] = args.sentences_files
