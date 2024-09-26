"""Contains CustomDataset class that inherits Dataset from PyTorch."""

from data_preparation.vocab import Vocab

from pathlib import Path
import pickle
import pyconll
from torch.utils.data import Dataset

# def main(conf):
#     vocabulary = Vocab(conf)
#     dataset = CustomDataset(conf, vocabulary, conf['train_directory'], sentences_pickle="example_set.pickle")
#
#     print(f"length of word-index dict: {len(dataset.vocab.vocab['word-index'])}")
#     print(f"length of grammeme-index dict: {len(dataset.vocab.vocab['grammeme-index'])}")
#     print(f"length of char-index dict: {len(dataset.vocab.vocab['char-index'])}")


class CustomDataset(Dataset):
    """Loads fastText embeddings and CONLL-U sentences from files. It inherits Dataset class of PyTorch module.

    Args:
        conf (dict): Dictionary with configuration parameters.
        vocab (Vocab): Instance of class containing vocabulary.
        files (list): List of .conllu files strings.

    Attributes:
        sentences_pyconll: Sentences in CONLL-U format.
        sentences: List of lists of strings. Raw sentences.

    Examples:
        >>> dataset = CustomDataset(config, Vocab(config), config['train_files'])
        >>> print(dataset.vocab.vocab["index-word"][dataset[66][0][8][0]])
    """

    def __init__(self, conf, vocab, files):
        self.conf = conf
        self.vocab = vocab
        self.files = files
        self.sentences_pyconll = None
        self.sentences = []
        self.get_all_sentences()

    def __len__(self):
        """Returns the number of sentences in dataset."""

        return len(self.sentences)

    def __getitem__(self, index):
        """Returns indices of words, chars, and grammemes for a sentence with a given index."""
        words, labels = self.vocab.sentence_to_indices(self.sentences[index],
                                                       self.sentences_pyconll[index])
        return words, labels

    def get_all_sentences(self):
        """
        Loads sentences from .conllu files and stores them as pyconll sentences in self.sentences_pyconll,
        and as lists of strings in self.sentences.
        """

        print(f"Loading sentences for dataset")
        self.sentences_pyconll = pyconll.load.load_from_file(self.files[0])
        for file in self.files[1:]:
            self.sentences_pyconll = self.sentences_pyconll + self.load.load_from_file(file)

        # notice that self.sentences contains words with capitalization
        # it will be ignored later when __getitem__ is called
        for sentence in self.sentences_pyconll:
            words = []
            for word in sentence:
                if '.' not in word.id:
                    words += [word.form]
            self.sentences += [words]
