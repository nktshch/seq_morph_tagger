"""Handles vocabulary words and embeddings. Contains method sentence_to_indices which is used in dataset.py"""

from pathlib import Path
import pyconll
from collections import Counter
import pickle
import fasttext
import numpy as np


def get_vocab(conf, rewrite=False):
    if rewrite:
        print("Rewriting vocab")
        return Vocab(conf)
    elif Path(conf["vocab_file"]).exists():
        print("Loading vocab from file")
        with open(conf["vocab_file"], 'rb') as vf:
            return pickle.load(vf)
    else:
        print("Creating vocab")
        return Vocab(conf)


class Vocab:
    """Contains dictionary of dictionaries and embeddings of wordforms.

    Dictionary keys are "word-index", "index-word", "grammeme-index", "index-grammeme",
    "char-index", "index-char", "singleton-index", "index-singleton".
    Each of them corresponds to a dictionary that maps element to index or vice versa.

    Class also has method sentence_to_indices which transforms all words and grammemes in a sentence into indices. It is
    used by CustomDataset and method predict.

    Args:
        conf: Dictionary with configuration parameters.

    Attributes:
        vocab: Dictionary with all the dictionaries that map strings to indices or vice versa.
        embeddings: Embeddings of wordforms. By default, they are uninitialized and are used only for encoder __init__.
            During training, they are sampled from normal distribution for words out of fastText library.
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.vocab = {} # dictionary of dictionaries, main object of the class

        self.sentences_train = None
        self.words_count = 0 # total number of words in the train files
        self.grammemes_by_freq = {} # dictionary of grammemes and their numbers of occurrences
        self.categories_by_freq = {} # dictionary of categories and their number of occurrences
        self.sorting_order = {}

        self.create_vocab()
        self.embeddings = np.ndarray((len(self.vocab['word-index']), self.conf['word_embeddings_dimension']))
        with open(self.conf["vocab_file"], 'wb+') as f:
            pickle.dump(self, f)
        print(f"Saved vocab at {self.conf['vocab_file']}")

    def create_vocab(self):
        """
        Creates dictionary Vocab.vocab of dictionaries {index:element} and {element:index}
        where element can be wordform, grammeme, char or singleton.
        """

        assert self.conf['train_files'], f"Directory {self.conf['language']} doesn't contain train files!"

        # There is no way of creating empty sentences_train variable that will allow summing itself with
        # pyconll.unit.conll.Conll object. For this reason, we first consider only the first file in the list,
        # and then add other sentences if there are any
        self.sentences_train = pyconll.load_from_file(self.conf['train_files'][0])
        for file in self.conf['train_files'][1:]:
            self.sentences_train = self.sentences_train + pyconll.load.load_from_file(file)

        self.vocab["word-index"], self.vocab["index-word"] = self.get_all_wordforms(self.sentences_train)
        self.vocab["grammeme-index"], self.vocab["index-grammeme"] = self.get_all_grammemes(self.sentences_train)
        self.vocab["char-index"], self.vocab["index-char"] = self.get_all_chars(self.sentences_train)
        self.vocab["singleton-index"], self.vocab["index-singleton"] = self.get_all_singletons(self.sentences_train)
        self.get_sorting_order()

    def create_embeddings(self, ft=None, dimension=300):
        """Loads embeddings and stores them in the class variable as list of ndarrays.

        If a word doesn't have the embedding, it is assigned a random one using normal distribution.

        Args:
            ft: Preloaded in main.py fastText library.
            dimension (int, default 300): The dimension of embeddings.
        """

        if ft is None:
            print("Loading fastText")
            ft = fasttext.load_model(self.conf['pretrained_embeddings'])

        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab['word-index'])),
                                           size=(len(self.vocab['word-index']), dimension))

        for word in ft.words:
            if word.lower() in self.vocab["word-index"].keys():
                self.embeddings[self.vocab["word-index"][word.lower()]] = ft[word]

    def get_all_wordforms(self, sentences):
        """
        Gets all wordforms in the dataset and creates two dictionaries:
        one with wordform:index pairs, other with index:wordform pairs.

        Args:
            sentences (pyconll.unit.conll.Conll): All of the sentences from which to get wordforms.

        Returns:
            tuple: Dictionaries with wordform->index and index->wordform pairs.
        """

        wordforms = set()
        for sentence in sentences:
            for _, token in enumerate(sentence):
                if '.' not in token.id and '-' not in token.id:
                    self.words_count += 1
                    if token.form.isdigit():
                        wordforms.add(self.conf['NUM'])
                    else:
                        wordforms.add(token.form.lower())
                    # wordforms.add(token.form)
        wordforms.add(self.conf['NUM'])
        wordforms = [self.conf['PAD'], self.conf['UNK']] + sorted(list(wordforms))
        return get_dictionaries(wordforms)

    def get_all_grammemes(self, sentences):
        """
        Gets all grammemes in the dataset and creates two dictionaries:
        one with grammeme:index pairs, other with index:grammeme pairs.
        Also, sorts all grammemes in vocab by frequency and stores them in a list. Creates defaultdict object
        that maps strings representing grammemes to indices in this list.
        This defaultdict is used when 'frequency' is chosen as grammeme order in config.
        During validation, test, and inference, if the grammeme is not in the sorted list,
        it will be assigned index equal to the length of the list (less frequent than all of them, in a sense).

        Args:
            sentences (pyconll.unit.conll.Conll): All of the sentences from which to get grammemes.

        Returns:
            tuple: Dictionaries with grammeme->index and index->grammeme pairs.
        """

        grammemes = set()
        categories = set()
        from collections import defaultdict
        grammemes_frequencies = defaultdict(int)
        categories_frequencies = defaultdict(int)
        for sentence in sentences:
            for _, token in enumerate(sentence):
                if '.' not in token.id and '-' not in token.id:
                    if token.upos is not None:
                        categories.add("POS")
                        categories_frequencies["POS"] += 1
                        grammemes.add("POS=" + token.upos)
                        grammemes_frequencies["POS=" + token.upos] += 1
                    for key in token.feats:
                        categories.add(key)
                        for feat in token.feats[key]:
                            categories_frequencies[key] += 1
                            grammemes.add(key + "=" + feat)
                            grammemes_frequencies[key + "=" + feat] += 1

        categories = list(categories)
        if self.conf['same_length'] is True:
            for cat in categories:
                grammemes_frequencies[cat + "=None"] = self.words_count - categories_frequencies[cat]
                grammemes.add(cat + "=None")
        grammemes = [self.conf["PAD"], self.conf["SOS"], self.conf["EOS"], self.conf["UNK"]] + sorted(list(grammemes))
        self.grammemes_by_freq = dict(sorted(grammemes_frequencies.items(), key=lambda item: item[1], reverse=True))
        self.categories_by_freq = dict(sorted(categories_frequencies.items(), key=lambda item: item[1], reverse=True))
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
                if '.' not in token.id and '-' not in token.id:
                    wordforms.add(token.form)
        wordforms = list(wordforms)        
        
        chars = set()
        for words in wordforms:
            chars.update(words)
        chars = [self.conf["PAD"]] + [self.conf["UNK"]] + sorted(list(chars))
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

    def get_sorting_order(self):
        """Sorts grammemes according to config, returns dictionary that will be used in sentence_to_indices()."""

        # there are actually 10 unique orders for xe loss because POS category is the most frequent

        grammemes = [g for g in self.vocab["grammeme-index"].keys() if "$" not in g]
        if self.conf['order'] == 'direct': # [1st]
            sorting_order = [g for g in grammemes if "POS" in g] + [g for g in grammemes if "POS" not in g]

        elif self.conf['order'] == 'reverse' and self.conf['pos_first'] is False: # [2nd]
            sorting_order = ([g for g in grammemes if "POS" in g] + [g for g in grammemes if "POS" not in g])[::-1]
        elif self.conf['order'] == 'reverse' and self.conf['pos_first'] is True: # [3rd]
            sorting_order = [g for g in grammemes if "POS" in g][::-1] + [g for g in grammemes if "POS" not in g][::-1]

        elif self.conf['order'] == 'frequency' and self.conf['sort_by'] == "grammemes":
            if self.conf['pos_first'] is False: # [4th]
                sorting_order = list(self.grammemes_by_freq.keys())
            else: # (pos_first is True) [5th]
                sorting_order = [g for g in list(self.grammemes_by_freq.keys()) if "POS" in g] + \
                                [g for g in list(self.grammemes_by_freq.keys()) if "POS" not in g]
        elif self.conf['order'] == 'frequency' and self.conf['sort_by'] == "categories": # [6th]
            sorting_order = sorted(grammemes,
                                   key=lambda i: self.categories_by_freq[i.split("=")[0]],
                                   reverse=True)

        elif self.conf['order'] == 'reverse_frequency' and self.conf['sort_by'] == "grammemes":
            if self.conf['pos_first'] is False: # [7th]
                sorting_order = list(self.grammemes_by_freq.keys())[::-1]
            else: # (pos_first is True) [8th]
                sorting_order = [g for g in list(self.grammemes_by_freq.keys()) if "POS" in g][::-1] + \
                                [g for g in list(self.grammemes_by_freq.keys()) if "POS" not in g][::-1]
        elif self.conf['order'] == 'reverse_frequency' and self.conf['sort_by'] == "categories":
            if self.conf['pos_first'] is False: # [9th]
                sorting_order = sorted(grammemes,
                                       key=lambda i: self.categories_by_freq[i.split("=")[0]])
            else: # (pos_first is True) [10th]
                if_pos_were_false = sorted(grammemes,
                                           key=lambda i: self.categories_by_freq[i.split("=")[0]])
                sorting_order = [g for g in if_pos_were_false if "POS" in g] + \
                                [g for g in if_pos_were_false if "POS" not in g]
        else:
            raise ValueError(f"Unknown order of grammemes: {self.conf['order']}, {self.conf['pos_first']}")
        self.sorting_order = {g: i for i, g in enumerate(sorting_order)}

    def sentence_to_indices(self, sentence, sentence_pyconll=None, training=True,
                            unk_word_id=1, unk_char_id=1, unk_grammeme_id=3):
        """Returns indices of words, chars, and grammemes for a sentence. Used by CustomDataset and in predict.

        It uses vocab dictionaries to map words, chars, and grammemes ot indices.
        Words are lowered before assignment.
        If an unknown element is met, corresponding id is used instead. These ids are not passed and should be assigned
        according to methods get_all_grammemes, get_all_chars, and create_vocab (see the line where
        self.vocab_wordforms is created).

        Args:
            sentence (list): Sentence as a list of strings. Used for words and chars.
            sentence_pyconll (pyconll.unit.conll.Conll, default None): Sentence in pyconll format. Used for grammemes.
                During inference, there will be no pyconll sentences, which means that method should return list
                of n_words empty lists as labels.
            training (bool, default True): If True, singletons will randomly be substituted by unk token.
            unk_word_id (int, default 1): The id of the unk word token in dictionary.
            unk_char_id (int, default 1): The id of the unk char token in dictionary.
            unk_grammeme_id (int, default 3): The id of the unk grammeme token in dictionary.
        """

        words = []
        labels = []
        for word in sentence:
            if word.isdigit():
                word_ids = [self.vocab["word-index"][self.conf['NUM']]]
            elif training is True and word.lower() in self.vocab["singleton-index"].keys() and np.random.rand() < self.conf["singleton_substitution"]:
                word_ids = [self.vocab["word-index"][self.conf['UNK']]]
            else:
                word_ids = [self.vocab["word-index"].get(word.lower(), unk_word_id)]

            for char in word:
                word_ids += [self.vocab["char-index"].get(char, unk_char_id)]
            words += [word_ids]

        if sentence_pyconll:
            for word in sentence_pyconll:
                if '.' not in word.id and '-' not in word.id:
                    grammeme_strings = set()
                    category_strings = set()
                    if word.upos is not None:
                        grammeme_strings.add("POS=" + word.upos)
                        category_strings.add("POS")
                    grammeme_strings.update([key + "=" + feat for key in list(word.feats) for feat in list(word.feats[key])])
                    category_strings.update([key for key in list(word.feats)])
                    if self.conf['same_length'] is True:
                        grammeme_strings.update([c + "=None" for c in self.categories_by_freq.keys()
                                                 if c not in category_strings])
                    grammeme_strings = sorted(list(grammeme_strings),
                                              key=lambda item: self.sorting_order.get(item, len(self.sorting_order)))
                    grammeme_ids = [self.vocab["grammeme-index"].get(g, unk_grammeme_id) for g in grammeme_strings]
                    labels += [grammeme_ids]

        else:
            for _ in sentence:
                labels += [[]]

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


