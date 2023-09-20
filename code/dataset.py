"""
Docstring for dataset.py
"""

# site: papers with code 
import os
import pickle
import pyconll
import argparse
import torch
from torch.utils.data import Dataset

import numpy as np

import fasttext
import fasttext.util

from vocab import Vocab


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument('train_files', nargs='+')
    argp.add_argument('--sentences_files', nargs='+')

    args = argp.parse_args()
    return args


def main(args):
    print(args)

    dataset = CustomDataset(args)

    print(len(dataset.vocab.vocab["word-index"]))
    print(len(dataset.vocab.vocab["grammeme-index"]))
    print(len(dataset.vocab.vocab["char-index"]))
    print(dataset.sentences[66])


    # dataset is needed to load raw sentences. It has __len__ and __getitem__
    # dataloader has access to dataset and to vocab. It creates tensors for tokens, chars, features. It works with paddings for
    # words, grammemes, chars. It can be done in CustomDataset

    # Vocab is always the same (uses train)

    # we assume that we know grammemes and other stuff if we work with dataset
    # if we don't, we have to handle them separately in a different method -  function predict() that doesn't use CustomDataset

    # get to know Dataloader

class CustomDataset(Dataset):
    def __init__(self, args):  # data_dir and data_type because we have to use train or dev or test. vocab is created here
        self.vocab = Vocab(args.train_files)
        self.sentences = []
        self.get_all_sentences(args.sentences_files)
        if (os.path.exists("embeddings.npz")):
            print("embeddings.npz found")
            with np.load("embeddings.npz") as npz:
                self.embeddings = npz["embeddings"]
        else:
            print("embeddings.npz not found")
            self.create_embeddings()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def get_all_sentences(self, files):
        print("Loading sentences")

        if (os.path.exists("sentences.pickle")):
            print("sentencese.pickle found")
            with open("sentences.pickle", 'rb') as f:
                self.sentences = pickle.load(f)
        else:
            print("sentencese.pickle not found")
            for file in files:
                sentences_pyconll = pyconll.load.load_from_file(file)
                self.sentences = self.sentences + [sentence.text for sentence in sentences_pyconll]
            with open("sentences.pickle", 'wb') as f:
                pickle.dump(self.sentences, f)


    def create_embeddings(self, dimension=300):  # fastText instead
        print("Loading fastText embeddings")
        fasttext.util.download_model('ru', if_exists='ignore')
        ft = fasttext.load_model('cc.ru.300.bin')

        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab.vocab['word-index'])), size=(len(self.vocab.vocab['word-index']), dimension))

        total = 0
        for word in ft.get_words():
            if word in self.vocab.vocab["word-index"].keys():
                total += 1
                self.embeddings[self.vocab.vocab["word-index"][word]] = ft[word]
            else:
                pass
                # self.embeddings[word] = np.random.normal(scale=2.0 / (dimension + len(self.vocab.vocab['word-index'])), size=(len(self.vocab.vocab['word-index']), 1))
        print(f"{total} of {len(self.vocab.vocab['word-index'])} had pretrained embeddings")

        np.savez_compressed("embeddings.npz", embeddings=self.embeddings)
        print("Saved embeddings")

        # initialize random embedding in case a word doesn't have fastText vector
        # embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab['word-index'])), size=(len(self.vocab['word-index']), dimension))
        # total = 0
        # with open(in_filename, encoding="utf-8") as f:
        #     for line in f:
        #         line = line.strip().split(" ")
        #         word = line[0]
        #         # reassign values if a word actually has pretrained vector
        #         if word in self.vocab["word-index"].keys():
        #             index = self.vocab["word-index"][word]
        #             embedding = [float(x) for x in line[1:dimension + 1]]
        #             embeddings[index] = np.asarray(embedding)
        #             total += 1
        # print(f"{total} of {len(self.vocab['word-index'])} words had embeddings")

        # self.embeddings = embeddings
        # np.savez_compressed("embeddings.npz", embeddings=embeddings)
        # print("Saved embeddings")


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
