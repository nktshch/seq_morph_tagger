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


def main(conf):
    print(conf)

    dataset = CustomDataset(conf)

    print(f"length of word-index dict: {len(dataset.vocab.vocab['word-index'])}")
    print(f"length of grammeme-index dict: {len(dataset.vocab.vocab['grammeme-index'])}")
    print(f"length of char-index dict: {len(dataset.vocab.vocab['char-index'])}")
    print(dataset.vocab.vocab["index-word"][dataset[66][0][8][0]])
    for grammeme in dataset[66][1][8]:
        print(dataset.vocab.vocab["index-grammeme"][grammeme])

    # dataset is needed to load raw sentences. It has __len__ and __getitem__
    # dataloader has access to dataset and to vocab. It creates tensors for tokens, chars, features. It works with paddings for
    # words, grammemes, chars. It can be done in CustomDataset

    # we assume that we know grammemes and other stuff if we work with dataset
    # if we don't, we have to handle them separately in a different method -  function predict() that doesn't use CustomDataset

class CustomDataset(Dataset):
    """
    docstring for CustomDataset
    """
    def __init__(self, conf):
        self.vocab = Vocab(conf["train_files"])
        self.sentences_pyconll = None
        self.sentences = []
        self.get_all_sentences(conf["sentences_files"])
        self.embeddings = []
        self.get_all_embeddings(dimension=conf['embeddings_dimension'])

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
        print("Loading sentences")

        if (os.path.exists("sentences.pickle")):
            with open("sentences.pickle", 'rb') as f:
                self.sentences_pyconll = pickle.load(f)
        else:
            print("There is no file containing sentences")
            self.sentences_pyconll = self.sentences_pyconll + pyconll.load.load_from_file(files[0])
            for file in files[1:]:
                self.sentences_pyconll = pyconll.load.load_from_file(file)
            with open("sentences.pickle", 'wb') as f:
                pickle.dump(self.sentences_pyconll, f)
                print("Saved sentences")
        for sentence in self.sentences_pyconll:
            words = []
            for word in sentence:
                words += [word.form]
            self.sentences += [words]

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


if __name__ == "__main__":
    configurate()
    main(config)
