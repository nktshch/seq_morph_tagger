"""
Docstring for dataset.py
"""

import os
import pickle
import pyconll
import numpy as np
import torch
from torch.utils.data import Dataset
import fasttext
import fasttext.util

from vocab import Vocab
from config import configurate
from config import config


def main(conf):
    print(conf)

    dataset = CustomDataset(conf)

    print(f"length of word-index dict: {len(dataset.vocab.vocab['word-index'])}")
    print(f"length of grammeme-index dict: {len(dataset.vocab.vocab['grammeme-index'])}")
    print(f"length of char-index dict: {len(dataset.vocab.vocab['char-index'])}")
    print(dataset.vocab.vocab["index-word"][dataset[66][0][8][0]])
    for grammeme in dataset[66][1][8]:
        print(dataset.vocab.vocab["index-grammeme"][grammeme])

    # we assume that we know grammemes and other stuff if we work with dataset
    # if we don't, we have to handle them separately in a different method -  function predict() that doesn't use CustomDataset

class CustomDataset(Dataset):
    """
    Class loads fastText embeddings and CONLL-U sentences from files. It inherits Dataset class of PyTorch module.
    __len__ returns the number of sentences.
    __getitem__ returns indices of words, chars, and grammemes for a sentence with a given index.

    Parameters
    ----------
    conf : dict
        Dictionary with configuration parameters
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.vocab = Vocab(self.conf)
        self.sentences_pyconll = None
        self.sentences = []
        self.get_all_sentences(self.conf["conllu_files"])
        self.embeddings = []
        self.get_all_embeddings(self.conf["embeddings_file"], dimension=self.conf['word_embeddings_dimension'])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        words = []
        labels = []
        for word in self.sentences[index]:
            word_ids = [self.vocab.vocab["word-index"][word]]
            for char in word:
                word_ids += [self.vocab.vocab["char-index"][char]]
            words += [word_ids]

        for word in self.sentences_pyconll[index]:
            grammeme_ids = []
            if word.upos is not None:
                grammeme_ids = [self.vocab.vocab["grammeme-index"]["POS=" + word.upos]]
            grammeme_ids += [self.vocab.vocab["grammeme-index"][key + "=" + feat] for key in word.feats for feat in word.feats[key]]
            labels += [grammeme_ids]
        return words, labels

    def get_all_sentences(self, files):
        """
        Loads sentences from .pickle file if there is one, and from .conllu files otherwise. In the second case, saves
        the sentences in the .pickle file.

        Parameters
        ----------
        files : list
            List of strings that are paths to .conllu files. Only used if there is no .pickle file
        """

        print("Loading sentences")

        if (os.path.exists(self.conf["sentences_pickle"])):
            with open(self.conf["sentences_pickle"], 'rb') as f:
                self.sentences_pyconll = pickle.load(f)
        else:
            print("There is no file containing pickle sentences")
            self.sentences_pyconll = pyconll.load.load_from_file(files[0])
            for file in files[1:]:
                self.sentences_pyconll = self.sentences_pyconll + self.load.load_from_file(file)
            with open(self.conf["sentences_pickle"], 'wb') as f:
                pickle.dump(self.sentences_pyconll, f)
                print("Saved sentences")
        for sentence in self.sentences_pyconll:
            words = []
            for word in sentence:
                words += [word.form]
            self.sentences += [words]

    def get_all_embeddings(self, file, dimension=300):
        """
        Loads embeddings from file and stores them in the class variable as list of ndarrays. If a word doesn't have
        the embedding, it is assigned a random one using normal distribution.

        Parameters
        ----------
        file : str
            The file containing fastText embeddings
        dimension : int, default 300
            The dimension of embeddings
        """

        print("Loading fastText embeddings")

        # fastText takes a lot of time to load embeddings (maybe there is no problem because we only load them once)
        assert os.path.exists(file), "There is no file containing embeddings"

        ft = fasttext.load_model(file)

        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab.vocab['word-index'])),
                                           size=(len(self.vocab.vocab['word-index']), dimension))

        total = 0
        for word in ft.get_words():
            if word in self.vocab.vocab["word-index"].keys():
                total += 1
                self.embeddings[self.vocab.vocab["word-index"][word]] = ft[word]
        print(f"{total} of {len(self.vocab.vocab['word-index'])} words had pretrained fastText embeddings")


if __name__ == "__main__":
    configurate()
    main(config)
