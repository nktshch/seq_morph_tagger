"""Pipeline of the model."""

from data_preparation.vocab import Vocab
from data_preparation.dataset import CustomDataset
from model.model import Model
from trainer import Trainer

import argparse
import json
import torch
from torch import cuda
import numpy as np


def configurate(config):
    """Adds command line arguments to config dictionary."""

    parse_arguments(config)

    config['device'] = 'cuda' if cuda.is_available() else 'cpu'
    config['word_LSTM_directions'] = 1 + int(config['word_LSTM_bidirectional'])
    config['char_LSTM_directions'] = 1 + int(config['char_LSTM_bidirectional'])
    config['grammeme_LSTM_hidden'] = config['word_LSTM_directions'] * config['word_LSTM_hidden']


def parse_arguments(config):
    argp = argparse.ArgumentParser()
    argp.add_argument('phase', help='train, test')
    argp.add_argument('train_directory', help='directory containing files with training data, they should be .conllu')
    argp.add_argument('valid_directory', help='directory containing files with validation data, they should be .conllu')
    argp.add_argument('test_directory', help='directory containing files with testing data, they should be .conllu')
    argp.add_argument('--model', help='file to save model to')
    argp.add_argument('--train_sentences_pickle',
                      help='file for (or containing) preloaded CoNLL-U training data, .pickle extension')
    argp.add_argument('--valid_sentences_pickle',
                      help='file for (or containing) preloaded CoNLL-U validation data, .pickle extension')
    argp.add_argument('--test_sentences_pickle',
                      help='file for (or containing) preloaded CoNLL-U testing data, .pickle extension')
    argp.add_argument('--dictionary_file',
                      help='file where all the dictionaries will be stored, .pickle extension')
    argp.add_argument('--embeddings_file',
                      help='file with word embeddings')

    args = argp.parse_args()
    config['phase'] = args.phase
    config['train_directory'] = args.train_directory
    config['valid_directory'] = args.valid_directory
    config['test_directory'] = args.test_directory
    config['model'] = args.model
    config['train_sentences_pickle'] = args.train_sentences_pickle
    config['valid_sentences_pickle'] = args.valid_sentences_pickle
    config['test_sentences_pickle'] = args.test_sentences_pickle
    config['dictionary_file'] = args.dictionary_file
    config['embeddings_file'] = args.embeddings_file


def main():
    with open('code/configs/config.json', 'r') as json_file:
        conf = json.load(json_file)
    configurate(conf)

    # for key in conf:
    #     print(f"{key} : {conf[key]}")

    print(f"Training model {conf['number_of_runs']} time(s)")
    for run_number in range(conf['number_of_runs']):
        torch.backends.cudnn.deterministic = True
        torch.random.manual_seed(run_number)
        torch.cuda.manual_seed(run_number)
        np.random.seed(run_number)

        vocabulary = Vocab(conf)
        train_data = CustomDataset(conf, vocabulary, conf['train_directory'],
                                   sentences_pickle=conf['train_sentences_pickle'], training_set=True)
        valid_data = CustomDataset(conf, vocabulary, conf['valid_directory'],
                                   sentences_pickle=conf['valid_sentences_pickle'], training_set=False)
        test_data = CustomDataset(conf, vocabulary, conf['test_directory'],
                                  sentences_pickle=conf['test_sentences_pickle'], training_set=False)
        model = Model(conf, train_data).to(conf['device'])
        trainer = Trainer(conf, model, valid_data, test_data, run_number=run_number, subset_size=10).to(conf['device'])
        trainer.epoch_loops()

        print("Training complete")


if __name__ == "__main__":
    main()
