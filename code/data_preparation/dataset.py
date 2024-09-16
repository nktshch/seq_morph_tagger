"""Docstring for dataset.py."""

from data_preparation.vocab import Vocab

from pathlib import Path
import pickle
import pyconll
import numpy as np
from torch.utils.data import Dataset
import fasttext
import fasttext.util


def main(conf):
    # print(conf)
    vocabulary = Vocab(conf)
    dataset = CustomDataset(conf, vocabulary, conf['train_directory'], sentences_pickle="example_set.pickle")

    print(f"length of word-index dict: {len(dataset.vocab.vocab['word-index'])}")
    print(f"length of grammeme-index dict: {len(dataset.vocab.vocab['grammeme-index'])}")
    print(f"length of char-index dict: {len(dataset.vocab.vocab['char-index'])}")

    # we assume that we know grammemes and other stuff if we work with dataset
    # if we don't, we have to handle them separately in a different method -  function predict() that
    # doesn't use CustomDataset


class CustomDataset(Dataset):
    """Loads fastText embeddings and CONLL-U sentences from files. It inherits Dataset class of PyTorch module.

    Args:
        conf (dict): Dictionary with configuration parameters.
        vocab (Vocab): Instance of class containing vocabulary.
        directory (string): Directory containing .conllu files. This parameter is used only if there is no .pickle file
            containing sentences.
        sentences_pickle (str, default None): Path to the .pickle file with sentences.
            If the file does not exist, class creates it. If None, does not save sentences in a file.
        training_set (bool, default True): Flag to show whether this is a training dataset. Creation of the embeddings
            depends on this.

    Attributes:
        vocab: Vocabulary created in vocab.py.
        embeddings: For training set, contains embeddings.
        sentences: List of lists of strings. Raw sentences.

    Examples:
        >>> dataset = CustomDataset(config, Vocab(config), config['train_files'], sentences_pickle="example_set.pickle")
        >>> print(dataset.vocab.vocab["index-word"][dataset[66][0][8][0]])
    """

    def __init__(self, conf, vocab, directory, sentences_pickle=None, training_set=True):
        self.conf = conf
        self.vocab = vocab
        self.directory = directory
        self.sentences_pickle = sentences_pickle
        self.training_set = training_set
        self.sentences_pyconll = None
        self.sentences = []
        self.get_all_sentences()
        if training_set:
            self.embeddings = []
            self.get_all_embeddings(self.conf["embeddings_file"], dimension=self.conf['word_embeddings_dimension'])

    def __len__(self):
        """Returns the number of sentences in dataset."""

        return len(self.sentences)

    def __getitem__(self, index):
        """Returns indices of words, chars, and grammemes for a sentence with a given index."""
        words, labels = \
            self.vocab.sentence_to_indices(self.sentences[index], self.sentences_pyconll[index], self.training_set)
        return words, labels

    def get_all_sentences(self):
        """Loads sentences from their .pickle file, if it exists.

        Otherwise, loads them from .conllu files and stores in .pickle file, if it is given as arguments.
        Also, stores the sentences as list of lists of words (strings).
        """

        print("Loading sentences for dataset")
        if self.sentences_pickle is not None:
            if Path(self.sentences_pickle).exists():
                with open(self.sentences_pickle, 'rb') as f:
                    self.sentences_pyconll = pickle.load(f)
            else:
                print(f"{self.sentences_pickle} does not exist")
                files = list(Path(self.directory).iterdir())
                self.sentences_pyconll = pyconll.load.load_from_file(files[0])
                for file in files[1:]:
                    self.sentences_pyconll = self.sentences_pyconll + self.load.load_from_file(file)

                with open(self.sentences_pickle, 'wb') as f:
                    pickle.dump(self.sentences_pyconll, f)
                    print(f"Saved sentences to {self.sentences_pickle}")
        else:
            print(".pickle file was not provided")
            files = list(Path(self.directory).iterdir())
            self.sentences_pyconll = pyconll.load.load_from_file(files[0])
            for file in files[1:]:
                self.sentences_pyconll = self.sentences_pyconll + self.load.load_from_file(file)

        for sentence in self.sentences_pyconll:
            words = []
            for word in sentence:
                words += [word.form]
            self.sentences += [words]

    def get_all_embeddings(self, file, dimension=300):
        """Loads embeddings from file and stores them in the class variable as list of ndarrays.

        If a word doesn't have the embedding, it is assigned a random one using normal distribution.

        Args:
            file (str): The file containing fastText embeddings.
            dimension (int, default 300): The dimension of embeddings.
        """

        print("Loading fastText embeddings")

        # fastText takes a lot of time to load embeddings (maybe there is no problem because we only load them once)
        assert Path(file).exists(), f"There is no file containing embeddings {file}"

        ft = fasttext.load_model(file)

        # the author of the original code has this scale
        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab.vocab['word-index'])),
                                           size=(len(self.vocab.vocab['word-index']), dimension))

        total = 0
        for word in ft.get_words():
            if word in self.vocab.vocab["word-index"].keys():
                total += 1
                self.embeddings[self.vocab.vocab["word-index"][word]] = ft[word]
        print(f"{total} of {len(self.vocab.vocab['word-index'])} words had pretrained fastText embeddings")
