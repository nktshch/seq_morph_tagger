"""
Docstring for vocab.py
"""

import os
import pyconll
from collections import Counter
import pickle


class Vocab:
    """
    Class contains dictionary of dictionaries Vocab.vocab
    Its keys are "word-index", "index-word", "grammeme-index", "index-grammeme", 
    "char-index", "index-char", "singleton-index", "index-singleton"
    Each of them corresponds to a dictionary that maps element to index or vice versa
    
    Parameters
    ----------
    conf : dict
        Dictionary with configuration parameters
    """
    
    def __init__(self, conf):

        self.train_files = conf["train_files"]
        self.dictionary_file = conf["dictionary_file"]
        self.PAD = conf["PAD"]
        self.EOS = conf["EOS"]
        self.SOS = conf["SOS"]
        self.NUM = conf["NUM"]
        self.UNK = conf["UNK"]
        self.NONE = conf["NONE"]
        self.vocab = {}
        self.dictionaries = ["word-index", "index-word", "grammeme-index", "index-grammeme", 
                             "char-index", "index-char", "singleton-index", "index-singleton"]

        self.generate_dictionaries()
                
    def generate_dictionaries(self):
        """
        Loads dictionary of dictionaries (vocab["word-index"] etc.) from dictionaries.pickle or
        creates it and saves into file 
        """
        
        print("Creating vocab")
        
        if os.path.exists(self.dictionary_file):
            with open(self.dictionary_file, 'rb') as f:
                self.vocab = pickle.load(f)
                if not len(self.vocab) == len(self.dictionaries):
                    print("File does not contain all dictionaries")
                    self.create_vocab()
                else:
                    pass
        else:
            print("There is no file containing dictionaries")
            self.create_vocab()
    
    def create_vocab(self):
        """
        Creates dictionary Vocab.vocab of dictionaries {index:element} and {element:index}
        where element is wordform, grammeme, char, singleton.
        This function also saves it into a file via pickle package
        """
        # There is no way to create empty sentences_train object that will allow summing itself with
        # pyconll.unit.conll.Conll object. For this reason we first consider only the first file in the list,
        # and then add another sentences if there are any
        sentences_train = pyconll.load_from_file(self.train_files[0])
        for file in self.train_files[1:]:
            sentences_train = sentences_train + pyconll.load.load_from_file(file)

        self.vocab["word-index"], self.vocab["index-word"] = self.get_all_wordforms(sentences_train)
        self.vocab["grammeme-index"], self.vocab["index-grammeme"] = self.get_all_grammemes(sentences_train)
        self.vocab["char-index"], self.vocab["index-char"] = self.get_all_chars(sentences_train)
        self.vocab["singleton-index"], self.vocab["index-singleton"] = self.get_all_singletons(sentences_train)
        
        with open(self.dictionary_file, 'wb') as f:
            pickle.dump(self.vocab, f)
        print("Saved dictionaries")
    
    def get_all_wordforms(self, sentences):
        """
        Gets all wordforms in the dataset and creates two dictionaries:
        one with wordform:index pairs, other with index:wordform pairs
        
        Parameters
        ----------
        sentences : pyconll.unit.conll.Conll
            All of the sentences from which to get wordforms, pyconll format
        Returns
        -------
        dict
            Dictionary with wordform:index pairs
        dict
            Dictionary with index:wordform pairs
        """
                
        wordforms = set()
        for sentence in sentences:
            for _, token in enumerate(sentence):
                wordforms.add(token.form)
        wordforms = list(wordforms)
        wordforms = [self.PAD, self.UNK] + wordforms
        return self.get_dictionaries(wordforms)
    
    def get_all_grammemes(self, sentences):
        """
        Gets all grammemes in the dataset and creates two dictionaries:
        one with grammeme:index pairs, other with index:grammeme pairs
        
        Parameters
        ----------
        sentences : pyconll.unit.conll.Conll
            All of the sentences from which to get grammemes, pyconll format
        Returns
        -------
        dict
            Dictionary with grammeme:index pairs
        dict
            Dictionary with index:grammeme pairs
        """
        
        grammemes = set()
        for sentence in sentences:
            for _, token in enumerate(sentence):
                if token.upos is not None:
                    grammemes.add("POS=" + token.upos)
                grammemes.update([key + "=" + feat for key in token.feats for feat in token.feats[key]])
        grammemes = list(grammemes)
        grammemes = [self.PAD, self.SOS, self.EOS, self.UNK] + grammemes
        return self.get_dictionaries(grammemes)
    
    def get_all_chars(self, sentences):
        """
        Gets all chars in the dataset and creates two dictionaries:
        one with char:index pairs, other with index:char pairs

        Returns
        -------
        dict
            Dictionary with char:index pairs
        dict
            Dictionary with index:char pairs
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
        return self.get_dictionaries(chars)
                
    def get_all_singletons(self, sentences):
        """
        Gets all singletons in the dataset and creates two dictionaries:
        one with singleton:index pairs, other with index:singleton pairs
        
        Parameters
        ----------
        sentences : pyconll.unit.conll.Conll
            All of the sentences from which to get singletons, pyconll format
        Returns
        -------
        dict
            Dictionary with singleton:index pairs
        dict
            Dictionary with index:singleton pairs
        """
        
        counter = Counter([token.form for sentence in sentences for _, token in enumerate(sentence)])
        singletons = [token for token, cnt in counter.items() if cnt == 1]
        return self.get_dictionaries(singletons)

    @staticmethod
    def get_dictionaries(data):
        """
        Create two dictionaries from list:
        first with element:index pairs, second with index:element pairs
        
        Parameters
        ----------
        data : list
            List with elements to be turned into dictionaries
        Returns
        -------
        dict
            Dictionary with element:index pairs
        dict
            Dictionary with index:element pairs
        """
        stoi = {element: index for index, element in enumerate(data)}
        itos = {index: element for index, element in enumerate(data)}
        return stoi, itos
