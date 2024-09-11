"""Contains dictionary with all necessary hyperparameters as well as paths to necessary files."""

import argparse
from torch import cuda


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
