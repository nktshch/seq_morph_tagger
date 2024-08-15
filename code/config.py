"""Contains dictionary with all necessary hyperparameters as well as paths to necessary files."""

import argparse
from torch import cuda

config = {'word_embeddings_dimension': 300,
          'char_embeddings_dimension': 100,
          'grammeme_embeddings_dimension': 150,
          'word_LSTM_bidirectional': True,
          'char_LSTM_bidirectional': True,
          'word_LSTM_hidden': 400,
          'char_LSTM_hidden': 150,
          'word_LSTM_input_dropout': 0.5,
          'word_LSTM_state_dropout': 0.3,
          'word_LSTM_output_dropout': 0.5,
          'grammeme_LSTM_input_dropout': 0.5,
          'clip': 5,
          'learning_rate': 1.0,
          'max_epochs': 400,
          'no_improv': 50,
          'bucket_train_data': True,
          'decoder_max_iterations': 12,
          'sentence_train_batch_size': 5,
          'sentence_eval_batch_size': 5,
          'UNK': "$UNK$",
          'NUM': "$NUM$",
          'NONE': "O",
          'SOS': "$SOS$",  # start of sequence
          'EOS': "$EOS$",  # end of sequence
          'PAD': "$PAD$",  # padding
          'embeddings_file': r".\data\cc.ru.300.bin"}  # file with fastText embeddings, .bin extension


def configurate():
    """Adds command line arguments to config dictionary."""

    parse_arguments()

    config['device'] = 'cuda' if cuda.is_available() else 'cpu'
    config['word_LSTM_directions'] = 1 + int(config['word_LSTM_bidirectional'])
    config['char_LSTM_directions'] = 1 + int(config['char_LSTM_bidirectional'])
    config['grammeme_LSTM_hidden'] = config['word_LSTM_directions'] * config['word_LSTM_hidden']


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument('phase', help='train, test')
    argp.add_argument('train_files', nargs='+', help='files with training data, .conllu extension')
    argp.add_argument('valid_files', nargs='+', help='files with validation data, .conllu extension')
    argp.add_argument('test_files', nargs='+', help='files with testing data, .conllu extension')
    argp.add_argument('--model', help='file to save save model to')
    argp.add_argument('--train_sentences_pickle',
                      help='file for (or containing) preloaded CoNLL-U training data, .pickle extension')
    argp.add_argument('--valid_sentences_pickle',
                      help='file for (or containing) preloaded CoNLL-U validation data, .pickle extension')
    argp.add_argument('--test_sentences_pickle',
                      help='file for (or containing) preloaded CoNLL-U testing data, .pickle extension')
    argp.add_argument('--dictionary_file',
                      help='file where all the dictionaries will be stored, .pickle extension')

    args = argp.parse_args()
    config['phase'] = args.phase
    config['train_files'] = args.train_files
    config['valid_files'] = args.valid_files
    config['test_files'] = args.test_files
    config['model'] = args.model
    config['train_sentences_pickle'] = args.train_sentences_pickle
    config['valid_sentences_pickle'] = args.valid_sentences_pickle
    config['test_sentences_pickle'] = args.test_sentences_pickle
    config['dictionary_file'] = args.dictionary_file
