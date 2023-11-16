"""
config.py contains dictionary with all necessary hyperparameters as well as paths to necessary files
"""

import argparse

config = {}

config['word_embeddings_dimension'] = 300 # fastText embeddings
config['char_embeddings_dimension'] = 100
config['char_LSTM_hidden'] = 150
config['word_LSTM_hidden'] = 400
config['word_LSTM_input_dropout'] = 0.5
config['grammeme_LSTM_hidden'] = 150
config['sentence_batch_size'] = 5
config['UNK'] = "$UNK$"
config['NUM'] = "$NUM$"
config['NONE'] = "O"
config['SOS'] = "$SOS$"
config['EOS'] = "$EOS$"
config['PAD'] = "$PAD$"



config['dictionary_file'] = ".\\data\\dictionaries.pickle" # file where all the dictionaries will be stored

def configurate():
    """
    Adds command line arguments to config dictionary.
    """

    parse_arguments()


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument('train_files', nargs='+')
    argp.add_argument('sentences_pickle', help='file for (or containing) preloaded CoNLL-U data, .pickle extension')
    argp.add_argument('--conllu_files', nargs='+')
    argp.add_argument('--embeddings_file', help='file with fastText embeddings, .bin extension')

    args = argp.parse_args()
    config['train_files'] = args.train_files
    config['sentences_pickle'] = args.sentences_pickle
    config['conllu_files'] = args.conllu_files
    config['embeddings_file'] = args.embeddings_file
