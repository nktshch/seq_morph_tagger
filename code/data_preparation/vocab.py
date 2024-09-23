"""Handles vocabulary words and embeddings. Conains method sentence_to_indices which is used in dataset.py"""

from pathlib import Path
import pyconll
from collections import Counter
import pickle
import fasttext
import numpy as np


class Vocab:
    """Contains dictionary of dictionaries and embeddings of wordforms.

    Its keys are "word-index", "index-word", "grammeme-index", "index-grammeme",
    "char-index", "index-char", "singleton-index", "index-singleton".
    Each of them corresponds to a dictionary that maps element to index or vice versa.

    Class also has method sentence_to_indices which transforms all words and grammemes in a sentence into indices. It is
    used by CustomDataset and method predict.

    Args:
        conf: Dictionary with configuration parameters.

    Attributes:
        vocab: Dictionary with all the dictionaries that map strings to indices or vice versa.
    """
    
    def __init__(self, conf):

        self.conf = conf
        self.vocab = {} # dictionary of dictionaries, main object of the class

        self.sentences_train = None
        self.set_train = None
        self.set_valid = None
        self.set_test = None
        self.set_pretrained = None

        self.vocab_wordforms = [] # all words that are in vocab
        self.pretrained_vocab_wordforms = [] # pretrained words that are in vocab
        self.embeddings = None # embeddings for words, used in encoder

        self.ft = None

        self.get_sets()
        # self.get_vocab(rewrite=True)
        self.create_vocab()
                
    # def get_vocab(self, rewrite=False):
    #     """ Loads dictionary of dictionaries (vocab["word-index"] etc.) from file or creates and saves it.
    #
    #     Args:
    #         rewrite (bool, default False): Tells to rewrite vocab even if it already exists.
    #     """
    #
    #     if rewrite:
    #         print("Rewriting vocab")
    #         self.create_vocab()
    #     elif Path(self.conf["vocab_file"]).exists():
    #         print("Loading vocab from file")
    #         with open(self.conf["vocab_file"], 'rb') as vf:
    #             self.vocab = pickle.load(vf)
    #             self.vocab_wordforms = list(self.vocab["word-index"].keys())
    #     else:
    #         print("Creating vocab")
    #         self.create_vocab()

    def get_sets(self):
        """
        Creates sets of wordforms that are in train, valid, and test sets.
        Also, gets all words that are in fastText library. All words are lowercase.
        """

        if not Path(self.conf["train_directory"]).is_dir() or not any(Path(self.conf["train_directory"]).iterdir()):
            raise NotADirectoryError(f"{self.conf['train_directory']} is not a valid directory for training files")

        # There is no way of creating empty sentences_train variable that will allow summing itself with
        # pyconll.unit.conll.Conll object. For this reason, we first consider only the first file in the list,
        # and then add other sentences if there are any
        train_files = list(Path(self.conf["train_directory"]).iterdir())
        self.sentences_train = pyconll.load_from_file(train_files[0])
        for file in train_files[1:]:
            self.sentences_train = self.sentences_train + pyconll.load.load_from_file(file)

        if Path(self.conf["valid_directory"]).is_dir():
            valid_files = list(Path(self.conf["valid_directory"]).iterdir())
            if valid_files:
                sentences_valid = pyconll.load_from_file(valid_files[0])
                for file in valid_files[1:]:
                    sentences_valid = sentences_valid + pyconll.load.load_from_file(file)
            else:
                sentences_valid = []
        else:
            sentences_valid = []

        if Path(self.conf["test_directory"]).is_dir():
            test_files = list(Path(self.conf["test_directory"]).iterdir())
            if test_files:
                sentences_test = pyconll.load_from_file(test_files[0])
                for file in test_files[1:]:
                    sentences_test = sentences_test + pyconll.load.load_from_file(file)
            else:
                sentences_test = []
        else:
            sentences_test = []

        # lines below create sets of all words present in corresponding datasets
        # usage of lowercase is hardcoded, it is not an option
        self.set_train = self.get_all_wordforms(self.sentences_train)
        self.set_valid = self.get_all_wordforms(sentences_valid)
        self.set_test = self.get_all_wordforms(sentences_test)

        print("Loading pretrained embeddings")
        self.ft = fasttext.load_model(self.conf['pretrained_embeddings'])
        self.set_pretrained = set(map(lambda x: x.lower(), self.ft.get_words()))

        # vocab contains all train words + valid and test words that have pretrained embeddings + special strings
        set_wordforms = self.set_train | (self.set_valid & self.set_pretrained) | (self.set_test & self.set_pretrained)
        set_wordforms.add(self.conf['NUM'])
        self.vocab_wordforms = [self.conf['PAD'], self.conf['UNK']] + sorted(list(set_wordforms))

        # these are words that are in vocab and have pretrained embeddings
        self.pretrained_vocab_wordforms = list((self.set_train | self.set_valid | self.set_test) & self.set_pretrained)

        print(f"{len(self.pretrained_vocab_wordforms)} of {len(self.vocab_wordforms)} words from vocab have pretrained fastText embeddings")

    def create_vocab(self):
        """
        Creates dictionary Vocab.vocab of dictionaries {index:element} and {element:index}
        where element can be wordform, grammeme, char or singleton.
        """

        print("Creating dictionaries")
        self.vocab["word-index"], self.vocab["index-word"] = get_dictionaries(self.vocab_wordforms)

        # for other dictionaries, only train dataset is used, handle them differently
        self.vocab["grammeme-index"], self.vocab["index-grammeme"] = self.get_all_grammemes(self.sentences_train)
        self.vocab["char-index"], self.vocab["index-char"] = self.get_all_chars(self.sentences_train)
        self.vocab["singleton-index"], self.vocab["index-singleton"] = self.get_all_singletons(self.sentences_train)
        
        # with open(self.conf["vocab_file"], 'wb') as f:
        #     pickle.dump(self.vocab, f)
        # print("Saved vocab")

    def create_embeddings(self, dimension=300):
        """Loads embeddings and stores them in the class variable as list of ndarrays.

        If a word doesn't have the embedding, it is assigned a random one using normal distribution.

        Args:
            dimension (int, default 300): The dimension of embeddings.
        """

        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab_wordforms)),
                                           size=(len(self.vocab_wordforms), dimension))

        for word in self.pretrained_vocab_wordforms:
            self.embeddings[self.vocab["word-index"][word]] = self.ft[word]

    def get_all_wordforms(self, sentences):
        """
        Gets all wordforms in the dataset and returns set of them. All words are lowercase. If a word contains only
        digits, returns special num token instead.

        Args:
            sentences (pyconll.unit.conll.Conll): All of the sentences from which to get wordforms.

        Returns:
            set: Set of all wordforms in the dataset.
        """

        wordforms = set()
        for sentence in sentences:
            for _, token in enumerate(sentence):
                if token.form.isdigit():
                    wordforms.add(self.conf['NUM'])
                else:
                    wordforms.add(token.form.lower())
        return wordforms

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
        grammemes = sorted(list(grammemes))
        grammemes = [self.conf["PAD"], self.conf["SOS"], self.conf["EOS"], self.conf["UNK"]] + grammemes
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
        chars = sorted(list(chars))
        chars = [self.conf["PAD"]] + [self.conf["UNK"]] + chars
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

    def sentence_to_indices(self, sentence, sentence_pyconll,
                            unk_word_id=1, unk_char_id=1, unk_grammeme_id=3):
        """Returns indices of words, chars, and grammemes for a sentence. Used by CustomDataset and in predict.

        It uses vocab dictionaries to map words, chars, and grammemes ot indices.
        Words are lowered before assignment.
        If an unknown element is met, corresponding id is used instead. These ids are not passed and should be assigned
        according to methods get_all_grammemes, get_all_chars, and create_vocab (see the line where
        self.vocab_wordforms is created).

        Args:
            sentence (list): Sentence as a list of strings. Used for words and chars.
            sentence_pyconll (pyconll.unit.conll.Conll): Sentence in pyconll format. Used for grammemes.
            unk_word_id (int, default 1): The id of the unk word token in dictionary.
            unk_char_id (int, default 1): The id of the unk char token in dictionary.
            unk_grammeme_id (int, default 3): The id of the unk grammeme token in dictionary.

        """

        words = []
        labels = []
        for word in sentence:
            if word.isdigit():
                word_ids = [self.vocab["word-index"][self.conf['NUM']]]
            else:
                word_ids = [self.vocab["word-index"].get(word.lower(), unk_word_id)]
            for char in word:
                word_ids += [self.vocab["char-index"].get(char, unk_char_id)]
            words += [word_ids]

        for word in sentence_pyconll:
            grammeme_ids = []
            if word.upos is not None:
                grammeme_ids = [self.vocab["grammeme-index"]["POS=" + word.upos]]
            grammeme_ids += [
                self.vocab["grammeme-index"].get(key + "=" + feat, unk_grammeme_id) for key in word.feats for feat in word.feats[key]]
            labels += [grammeme_ids]

        return words, labels


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
