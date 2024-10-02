"""Pipeline of the model."""

from data_preparation.vocab import Vocab, get_vocab
from data_preparation.dataset import CustomDataset
from model.model import Model
from trainer import Trainer

import argparse
from pathlib import Path
import json
import fasttext
import torch
from torch import cuda
import numpy as np
import random


def parse_arguments():
    """Loads config from json file and adds command line arguments to it."""

    argp = argparse.ArgumentParser()
    argp.add_argument('config', help='json file containing configurations parameters')
    argp.add_argument('language', help='Directory with all files used for training. Must contain train conllu files')
    argp.add_argument('model', help='Directory with model and runs')
    argp.add_argument('--pretrained_embeddings',
                      help='file with word embeddings (fastText), .bin extension')

    args = argp.parse_args()

    with open(args.config, 'r') as json_file:
        config = json.load(json_file)

    if config['order'] not in ["direct", "reverse", "frequency"]:
        raise ValueError(f"Unknown order of grammemes: {config['order']}")

    config['language'] = args.language
    config['model'] = args.model
    config['pretrained_embeddings'] = args.pretrained_embeddings

    config['device'] = 'cuda' if cuda.is_available() else 'cpu'
    config['word_LSTM_directions'] = 1 + int(config['word_LSTM_bidirectional'])
    config['char_LSTM_directions'] = 1 + int(config['char_LSTM_bidirectional'])
    config['grammeme_LSTM_hidden'] = config['word_LSTM_directions'] * config['word_LSTM_hidden']

    config['train_files'] = [str(x) for x in Path(config['language']).glob("*train*.conllu")]
    config['valid_files'] = [str(x) for x in Path(config['language']).glob("*dev*.conllu")]
    config['test_files'] = [str(x) for x in Path(config['language']).glob("*test*.conllu")]
    config['vocab_file'] = config['model'] + "/vocab.pickle"

    Path(config['model']).mkdir(parents=True, exist_ok=True)
    return config


def main():
    conf = parse_arguments()

    # for key in conf:
    #     print(f"{key} : {conf[key]}")

    vocab = get_vocab(conf, rewrite=True)

    train_data = CustomDataset(conf, vocab, conf['train_files'])

    if conf['valid_files']:
        valid_data = CustomDataset(conf, vocab, conf['valid_files'])
    else:
        valid_data = None

    if conf['test_files']:
        test_data = CustomDataset(conf, vocab, conf['test_files'])
    else:
        test_data = None

    print(f"Training model {conf['number_of_runs']} time(s)")
    for run_number in range(conf['number_of_runs']):
        torch.backends.cudnn.deterministic = True
        torch.random.manual_seed(run_number)
        cuda.manual_seed(run_number)
        np.random.seed(run_number)
        random.seed(run_number)

        vocab.create_embeddings(dimension=conf['word_embeddings_dimension'])
        model = Model(conf, vocab).to(conf['device'])
        trainer = Trainer(conf, model, train_data, valid_data, test_data,
                          run_number=run_number).to(conf['device'])
        trainer.epoch_loops()

        print("Training complete")


if __name__ == "__main__":
    main()
