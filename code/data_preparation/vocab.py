"""Docstring for vocab.py."""

from pathlib import Path
import pyconll
from collections import Counter
import pickle
import fasttext
import numpy as np


class Vocab:
    """Contains dictionary of dictionaries.

    Its keys are "word-index", "index-word", "grammeme-index", "index-grammeme",
    "char-index", "index-char", "singleton-index", "index-singleton".
    Each of them corresponds to a dictionary that maps element to index or vice versa.

    Args:
        conf: Dictionary with configuration parameters.

    Attributes:
        vocab: Dictionary with all the dictionaries that map strings to indices or vice versa.
    """
    
    def __init__(self, conf):

        self.conf = conf
        self.train_directory = conf["train_directory"]
        self.valid_directory = conf["valid_directory"]
        self.test_directory = conf["test_directory"]
        self.vocab_file = conf["vocab_file"]
        self.embeddings_file = conf["embeddings_file"]
        self.PAD = conf["PAD"]
        self.EOS = conf["EOS"]
        self.SOS = conf["SOS"]
        self.NUM = conf["NUM"]
        self.UNK = conf["UNK"]
        self.NONE = conf["NONE"]
        self.vocab = {}
        self.dictionaries = ["word-index", "index-word", "grammeme-index", "index-grammeme", 
                             "char-index", "index-char", "singleton-index", "index-singleton"]

        self.pretrained_wordforms = []
        self.vocab_wordforms = []
        self.pretrained_vocab = []
        self.embeddings = None

        self.ft = None

        self.get_vocab_and_embeddings(rewrite=True)
                
    def get_vocab_and_embeddings(self, rewrite=False):
        """
        Loads dictionary of dictionaries (vocab["word-index"] etc.) and embeddings from files or
        creates and saves them.

        Args:
            rewrite (bool, default False): Tells to rewrite vocab and embeddings even if they already exist.
        """

        if rewrite:
            print("Rewriting vocab and embeddings")
            self.ft = fasttext.load_model(self.conf['pretrained_embeddings'])
            self.create_vocab()
            self.create_embeddings(self.conf['word_embeddings_dimension'])
        else:
            if Path(self.vocab_file).exists() and Path(self.embeddings_file).exists():
                print("Loading vocab and embeddings from files")
                with open(self.vocab_file, 'rb') as vf:
                    self.vocab = pickle.load(vf)
                with open(self.embeddings_file, 'rb') as ef:
                    self.embeddings = np.load(ef)
            elif Path(self.vocab_file).exists() and not Path(self.embeddings_file).exists():
                print("Loading vocab from file and creating embeddings")
                with open(self.vocab_file, 'rb') as vf:
                    self.vocab = pickle.load(vf)
                self.ft = fasttext.load_model(self.conf['pretrained_embeddings'])
                self.create_embeddings(self.conf['word_embeddings_dimension'])
            elif not Path(self.vocab_file).exists() and Path(self.embeddings_file).exists():
                print("Creating vocab and loading embeddings from file")
                with open(self.embeddings_file, 'rb') as ef:
                    self.embeddings = np.load(ef)
                self.ft = fasttext.load_model(self.conf['pretrained_embeddings'])
                self.create_vocab()
            else:
                print("Creating vocab and embeddings")
                self.ft = fasttext.load_model(self.conf['pretrained_embeddings'])
                self.create_vocab()
                self.create_embeddings(self.conf['word_embeddings_dimension'])
    
    def create_vocab(self):
        """
        Creates dictionary Vocab.vocab of dictionaries {index:element} and {element:index}
        where element is wordform, grammeme, char, singleton.
        This function also saves it into a file via pickle package.
        """

        # There is no way of creating empty sentences_train variable that will allow summing itself with
        # pyconll.unit.conll.Conll object. For this reason, we first consider only the first file in the list,
        # and then add other sentences if there are any
        train_files = list(Path(self.train_directory).iterdir())
        sentences_train = pyconll.load_from_file(train_files[0])
        for file in train_files[1:]:
            sentences_train = sentences_train + pyconll.load.load_from_file(file)

        valid_files = list(Path(self.valid_directory).iterdir())
        sentences_valid = pyconll.load_from_file(valid_files[0])
        for file in valid_files[1:]:
            sentences_valid = sentences_valid + pyconll.load.load_from_file(file)

        test_files = list(Path(self.test_directory).iterdir())
        sentences_test = pyconll.load_from_file(test_files[0])
        for file in test_files[1:]:
            sentences_test = sentences_test + pyconll.load.load_from_file(file)

        set_train = get_wordforms(sentences_train)
        set_valid = get_wordforms(sentences_valid)
        set_test = get_wordforms(sentences_test)
        set_pretrained = set(map(lambda x: x.lower(), self.ft.get_words()))
        self.pretrained_wordforms = list(set_pretrained)

        # vocab contains all train words and valid and test words that have pretrained embeddings
        set_wordforms = set_train | (set_valid & set_pretrained) | (set_test & set_pretrained)
        self.pretrained_vocab = list((set_train | set_valid | set_test) & set_pretrained)
        self.vocab_wordforms = [self.PAD, self.UNK] + list(set_wordforms)
        self.vocab["word-index"], self.vocab["index-word"] = get_dictionaries(self.vocab_wordforms)

        self.vocab["grammeme-index"], self.vocab["index-grammeme"] = self.get_all_grammemes(sentences_train)
        self.vocab["char-index"], self.vocab["index-char"] = self.get_all_chars(sentences_train)
        self.vocab["singleton-index"], self.vocab["index-singleton"] = self.get_all_singletons(sentences_train)
        
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self.vocab, f)
        print("Saved vocab")


    def create_embeddings(self, dimension=300):
        """Loads embeddings and stores them in the class variable as list of ndarrays.

        If a word doesn't have the embedding, it is assigned a random one using normal distribution.

        Args:
            file (str): The file containing fastText embeddings.
            dimension (int, default 300): The dimension of embeddings.
        """
        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab_wordforms)),
                                           size=(len(self.vocab_wordforms), dimension))

        for word in self.pretrained_vocab:
            self.embeddings[self.vocab["word-index"][word]] = self.ft[word]
        print(f"{len(self.pretrained_vocab)} of {len(self.vocab_wordforms)} words from vocab had pretrained fastText embeddings")

        with open(self.embeddings_file, 'wb') as f:
            np.save(f, self.embeddings)
        print("Saved embeddings")


    # def get_all_wordforms(self, sentences):
    #     """
    #     Gets all wordforms in the dataset and creates two dictionaries:
    #     one with wordform:index pairs, other with index:wordform pairs.
    #
    #     Args:
    #         sentences (pyconll.unit.conll.Conll): All of the sentences from which to get wordforms.
    #
    #     Returns:
    #         tuple: Dictionaries with wordform->index and index->wordform pairs.
    #     """
    #
    #     wordforms = set()
    #     for sentence in sentences:
    #         for _, token in enumerate(sentence):
    #             wordforms.add(token.form)
    #     wordforms = list(wordforms)
    #     wordforms = [self.PAD, self.UNK] + wordforms
    #     return self.get_dictionaries(wordforms)


    def get_all_grammemes(self, sentences):
        """
        Gets all grammemes in the dataset and creates two dictionaries:
        one with grammeme:index pairs, other with index:grammeme pairs.

        Args:
            sentences (pyconll.unit.conll.Conll): All of the sentences from which to get grammemes.

        Returns:
            tuple: Dictionaries with grammeme->index and index->grammeme pairs.
        """
        
        grammemes = set()
        for sentence in sentences:
            for _, token in enumerate(sentence):
                if token.upos is not None:
                    grammemes.add("POS=" + token.upos)
                grammemes.update([key + "=" + feat for key in token.feats for feat in token.feats[key]])
        grammemes = list(grammemes)
        grammemes = [self.PAD, self.SOS, self.EOS, self.UNK] + grammemes
        return get_dictionaries(grammemes)
    
    def get_all_chars(self, sentences):
        """
        Gets all chars in the dataset and creates two dictionaries:
        one with char:index pairs, other with index:char pairs.

        Args:
            sentences (pyconll.unit.conll.Conll): All of the sentences from which to get chars.

        Returns:
            tuple: Dictionaries with char->index and index->char pairs.
        """
        
        wordforms = set()
        for sentence in sentences:
            for _, token in enumerate(sentence):
                wordforms.add(token.form)
        wordforms = list(wordforms)        
        
        chars = set()
        for words in wordforms:
            chars.update(words)
        chars = list(chars)
        chars = [self.PAD] + [self.UNK] + chars
        return get_dictionaries(chars)


    def get_all_singletons(self, sentences):
        """
        Gets all singletons in the dataset and creates two dictionaries:
        one with singleton:index pairs, other with index:singleton pairs.

        Args:
            sentences (pyconll.unit.conll.Conll): All of the sentences from which to get singletons.

        Returns:
            tuple: Dictionaries with singleton->index and index->singleton pairs.
        """
        
        counter = Counter([token.form for sentence in sentences for _, token in enumerate(sentence)])
        singletons = [token for token, cnt in counter.items() if cnt == 1]
        return get_dictionaries(singletons)

    def sentence_to_indices(self, sentence, sentence_pyconll):
        """Returns indices of words, chars, and grammemes for a sentence. Used by CustomDataset and in predict."""

        words = []
        labels = []
        for word in sentence:
            word_ids = [self.vocab["word-index"].get(word.lower(), 1)]
            for char in word:
                word_ids += [self.vocab["char-index"].get(char, 1)]
            words += [word_ids]

        for word in sentence_pyconll:
            grammeme_ids = []
            if word.upos is not None:
                grammeme_ids = [self.vocab["grammeme-index"]["POS=" + word.upos]]
            grammeme_ids += [
                self.vocab["grammeme-index"][key + "=" + feat] for key in word.feats for feat in word.feats[key]]
            labels += [grammeme_ids]

        return words, labels


def get_wordforms(sentences):
    """
    Gets all wordforms in the dataset and returns set of them.

    Args:
        sentences (pyconll.unit.conll.Conll): All of the sentences from which to get wordforms.

    Returns:
        set: Set of all wordforms in the dataset.
    """

    wordforms = set()
    for sentence in sentences:
        for _, token in enumerate(sentence):
            wordforms.add(token.form.lower())
    return wordforms


def get_dictionaries(data):
    """
    Create two dictionaries from list:
    first with element:index pairs, second with index:element pairs.

    Args:
        data (list): List with elements to be turned into dictionaries.

    Returns:
        tuple: Dictionaries with element->index and index->element pairs.
    """

    stoi = {element: index for index, element in enumerate(data)}
    itos = {index: element for index, element in enumerate(data)}
    return stoi, itos
