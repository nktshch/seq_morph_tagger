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


def different_orders():
    # method is used for checking that orders are loaded as expected
    conf = parse_arguments()

    pos_first = [False, True]
    orders = ["direct", "reverse", "frequency", "reverse_frequency"]

    results = []
    for o in orders:
        for p in pos_first:
            conf['order'] = o
            conf['pos_first'] = p
            conf['vocab_file'] = f"{conf['model']}/order_test_vocab_{o}_{p}.pickle"

            vocab = get_vocab(conf, rewrite=True)
            data = CustomDataset(conf, vocab, ["./data/order_test.conllu"])

            (_, labels), sentence = data[0]
            tag_ids = labels[2]
            tag = [vocab.vocab["index-grammeme"][g] for g in tag_ids]
            results += [(tag, o, p)]

    for r in results:
        print(*r)


def parse_arguments():
    """Loads config from json file and adds command line arguments to it."""

    argp = argparse.ArgumentParser()
    argp.add_argument('config', help='json file containing configurations parameters')
    argp.add_argument('language', help='Directory with all files used for training. Must contain train conllu files')
    argp.add_argument('model', help='Directory with model and vocab files')
    argp.add_argument('--seed', help='Random seed. If specified, model will be run only once')
    argp.add_argument('--pretrained_embeddings',
                      help='file with word embeddings (fastText), .bin extension')
    # it is an optional argument even though program will crash if it is not specified
    # in the future, it should randomly initialize all embeddings if pretrained are not provided

    args = argp.parse_args()

    with open(args.config, 'r') as json_file:
        config = json.load(json_file)

    if config['order'] not in ["direct", "reverse", "frequency", "reverse_frequency"]:
        raise ValueError(f"Unknown order of grammemes: {config['order']}")

    if config['loss'] not in ["xe", "oaxe"]:
        raise ValueError(f"Unknown loss: {config['loss']}")

    config['teacher_forcing'] = False if config['loss'] == "oaxe" else True

    config['language'] = args.language
    config['model'] = args.model
    config['seed'] = int(args.seed) if args.seed is not None else None
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
        valid_data = CustomDataset(conf, vocab, conf['valid_files'], training=False)
        valid_set = valid_data.words_set
    else:
        valid_data = None
        valid_set = set()

    if conf['test_files']:
        test_data = CustomDataset(conf, vocab, conf['test_files'], training=False)
        test_set = test_data.words_set
    else:
        test_data = None
        test_set = set()

    if valid_data is not None or test_data is not None:
        print("Loading fastText")
        ft = fasttext.load_model(conf['pretrained_embeddings'])
        oov_pretrained_vocab = {}
        oov_pretrained_set = valid_set | test_set
        for word in ft.words:
            if word.lower() in oov_pretrained_set:
                oov_pretrained_vocab[word.lower()] = ft[word]
    else:
        ft = None
        oov_pretrained_vocab = None

    def run(seed):
        torch.backends.cudnn.deterministic = True
        torch.random.manual_seed(seed)
        cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        vocab.create_embeddings(ft=ft, dimension=conf['word_embeddings_dimension'])
        model = Model(conf, vocab).to(conf['device'])
        trainer = Trainer(conf, model, train_data, valid_data, test_data,
                          run_number=seed, subset_size=0).to(conf['device'])
        trainer.epoch_loops(oov_pretrained_vocab=oov_pretrained_vocab)

        print("Training complete")

    if conf['seed'] is None:
        print(f"Training model {conf['number_of_runs']} time(s)")
        for run_number in range(conf['number_of_runs']):
            run(run_number)
    else:
        print(f"Training model 1 time with seed {conf['seed']}")
        run(conf['seed'])


if __name__ == "__main__":
    main()
    # different_orders()
