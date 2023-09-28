"""
Docstring for dataset.py
"""

# site: papers with code 
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


def main(config):
    print(config)

    dataset = CustomDataset(config)

    print(f"length of word-index dict: {len(dataset.vocab.vocab['word-index'])}")
    print(f"length of grammeme-index dict: {len(dataset.vocab.vocab['grammeme-index'])}")
    print(f"length of char-index dict: {len(dataset.vocab.vocab['char-index'])}")
    print(dataset[66])

    # dataset is needed to load raw sentences. It has __len__ and __getitem__
    # dataloader has access to dataset and to vocab. It creates tensors for tokens, chars, features. It works with paddings for
    # words, grammemes, chars. It can be done in CustomDataset

    # we assume that we know grammemes and other stuff if we work with dataset
    # if we don't, we have to handle them separately in a different method -  function predict() that doesn't use CustomDataset

class CustomDataset(Dataset):
    """
    docstring for CustomDataset
    """
    def __init__(self, config):  # data_dir and data_type because we have to use train or dev or test. vocab is created here
        self.vocab = Vocab(config["train_files"])
        self.sentences = []
        self.get_all_sentences(config["sentences_files"])
        self.embeddings = []
        self.get_all_embeddings(dimension=config['embeddings_dimension'])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def get_all_sentences(self, files):
        print("Loading sentences")

        if (os.path.exists("sentences.pickle")):
            with open("sentences.pickle", 'rb') as f:
                self.sentences = pickle.load(f)
        else:
            print("There is no file containing sentences")
            for file in files:
                sentences_pyconll = pyconll.load.load_from_file(file)
                self.sentences = self.sentences + [sentence.text for sentence in sentences_pyconll]
            with open("sentences.pickle", 'wb') as f:
                pickle.dump(self.sentences, f)
                print("Saved sentences")

    def get_all_embeddings(self, dimension=300):
        print("Loading fastText embeddings")

        # fastText takes a lot of time to load embeddings (maybe there is no problem because we only load them once)
        if (os.path.exists("embeddings.npz")):
            with np.load("embeddings.npz") as npz:
                self.embeddings = npz["embeddings"]
        else:
            print("There is no file containing embeddings")
            fasttext.util.download_model('ru', if_exists='ignore')
            ft = fasttext.load_model('cc.ru.300.bin')

            self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab.vocab['word-index'])),
                                               size=(len(self.vocab.vocab['word-index']), dimension))

            total = 0
            for word in ft.get_words():
                if word in self.vocab.vocab["word-index"].keys():
                    total += 1
                    self.embeddings[self.vocab.vocab["word-index"][word]] = ft[word]
            print(f"{total} of {len(self.vocab.vocab['word-index'])} had pretrained fastText embeddings")

            np.savez_compressed("embeddings.npz", embeddings=self.embeddings)
            print("Saved embeddings")

    def parse_sentence(self, sentences, max_number):
        """
        Given sentences, for every word in a sentences creates list word[word index, char indices, grammeme indices],
        then creates list of such lists for every sentence and yields it. Word index (int), char indices (list), and
        grammeme indices (list) are created using ['word-index'], ['char-index'], and ['grammeme-index'] respectively.

        Parameters
        ----------
        sentences : pyconll.unit.conll.Conll
            All of the sentences that from which to create parsed versions
        max_number : int
            Maximum number of sentences that will be parsed
        """
        current_number = 0
        for sentence in sentences:
            words = []
            for _, token in enumerate(sentence):
                if token.form in self.vocab.vocab["word-index"].keys():
                    word_id = self.vocab.vocab["word-index"][token.form]
                else:
                    word_id = self.vocab.vocab["word-index"][UNK] # assign UNK if the word is not present in vocab

                chars_ids = []
                for char in token.form:
                    if char in self.vocab.vocab["char-index"].keys():
                        chars_ids += [self.vocab.vocab["char-index"][char]]
                    # we ignore chars that don't have embeddings

                grammemes_ids = []
                # POS tag always goes first in the list of grammemes
                if token.upos is not None and "POS=" + token.upos in self.vocab.vocab["grammeme-index"].keys():
                    grammemes_ids += [self.vocab.vocab["grammeme-index"]["POS=" + token.upos]]
                for key in token.feats:
                    for feat in token.feats[key]:
                        if key + "=" + feat in self.vocab.vocab["grammeme-index"].keys():
                            grammemes_ids += [self.vocab.vocab["grammeme-index"][key + "=" + feat]]

                word = [word_id, chars_ids, grammemes_ids]
                words += [word]
            yield words
            current_number += 1
            if current_number == max_number:
                break


if __name__ == "__main__":
    configurate()
    main(config)
