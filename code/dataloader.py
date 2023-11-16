"""
Docstring for dataloader.py
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import dataset
from config import configurate
from config import config

# all layers and classes should be descendants of nn.Module and all of them should have forward()
# multitask learning

def main(conf):
    # print(conf)
    model = Model(conf)
    loader = torch.utils.data.DataLoader(model.data, batch_size=conf['sentence_batch_size'], collate_fn=collate_batch)
    progress_bar = tqdm(enumerate(loader), disable=True)
    for _, (words_batch, chars_batch, labels_batch) in progress_bar:
        logits, loss = model(words_batch, chars_batch, labels_batch)

class WordEmbeddings(nn.Module):
    """
    Class creates embeddings for words and chars, puts them through LSTMs to produce word embeddings. It has access to
    the instance of CustomDataset, created by Model class.

    Method forward takes words_batch -- size (batch_size, max_sentence_length) -- and
    chars_batch -- size (batch_size * max_sentence_length, max_word_length) -- as an input. It returns output from the
    LSTM for every word in a sentence. The final shape of the output is (batch_size, max_sentence_length, 2 * word_LSTM_hidden).

    Parameters
    ----------
    conf : dict
        Dictionary with configuration parameters
    data : CustomDataset
        Class instance from dataset.py
    """

    def __init__(self, conf, data):
        super().__init__()
        self.conf = conf
        self.data = data
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(data.embeddings))
        self.char_embeddings = nn.Embedding(len(data.vocab.vocab['char-index']), self.conf["char_embeddings_dimension"])
        self.charLSTM = nn.LSTM(input_size=self.conf['char_embeddings_dimension'],
                                hidden_size=self.conf['char_LSTM_hidden'], bidirectional=True, batch_first=True)
        self.wordLSTM = nn.LSTM(input_size=(self.conf['word_embeddings_dimension'] + self.conf['char_LSTM_hidden'] * 2),
                                hidden_size=self.conf['word_LSTM_hidden'], bidirectional=True, batch_first=True)

    def forward(self, words_batch, chars_batch):
        """
        Takes batches of indices of words and chars and creates embeddings with LSTM.

        Parameters
        ----------
        words_batch : torch.Tensor
            Tensor of words indices for every word in a batch. Size (batch_size, max_sentence_length)
        chars_batch : torch.Tensor
            Tensor of chars indices for every word in a batch. Size (batch_size * max_sentence_length, max_word_length)
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, max_sentence_length, 2 * word_LSTM_hidden), containing embeddings for every word
            in a sentence
        """

        words = self.word_embeddings(words_batch)
        chars = self.char_embeddings(chars_batch)
        # words has shape (batch_size, max_sentence_length, word_embeddings_dimension)
        # chars has shape (batch_size * max_sentence_length, max_word_length, char_embeddings_dimension)
        _, (hn, cn) = self.charLSTM(chars)
        # hn has shape (2, batch_size * max_sentence_length, char_LSTM_hidden)
        # 2 because of bidirectional LSTM
        chars = hn.view(self.conf['sentence_batch_size'], -1, hn.shape[2] * 2)
        # chars has shape (batch_size, max_sentence_length, 2 * char_LSTM_hidden)
        words = torch.concat((words, chars), dim=2)
        output, _ = self.wordLSTM(words.float()) # for some reason, without .float(), RuntimeError is raised (expected type Double but found Float)
        # final shape is (batch_size, max_sentence_length, 2 * word_LSTM_hidden)
        # print(output.shape)
        return output

class Model(nn.Module): # for now, it is here, maybe move it elsewhere
    """
    Class takes batches produced by DataLoader and assignes embeddings to words using WordEmbeddings class.
    Creates an instance of CustomDataset.

    Parameters
    ----------
    conf : dict
        Dictionary with configuration parameters
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.data = dataset.CustomDataset(self.conf)
        self.word_embeddings = WordEmbeddings(self.conf, self.data)
        self.grammeme_embeddings = None

    def forward(self, words_batch, chars_batch, labels_batch):
        """
        Parameters
        ----------
        words_batch : torch.Tensor
            Tensor of words indices for every word in a batch. Size (batch_size, max_sentence_length)
        chars_batch : torch.Tensor
            Tensor of chars indices for every word in a batch. Size (batch_size * max_sentence_length, max_word_length)
        """

        words = self.word_embeddings(words_batch, chars_batch)
        logits = 0
        loss = 0
        return logits, loss
    # train - separate function that has instance of dataloader

    # model produces vector of size (n_grammemes) that will be compared to one-hot representation of the labels
    # 150 (grammeme_LSTM_hidden) is used in lstm
    # in sequential model, we find the index of max value and pass it forward

def collate_batch(batch, pad_id=0, eos_id=2): # do all preprocessing here
    """
    Function takes batch created with CustomDataset and performs padding. batch consists of indices of words,
    chars, and labels (grammemes). The returned batches are tensors.

    Parameters
    ---------
    batch : list
        A single batch to be collated. Batch consists of tuples (words, labels) generated by CustomDataset
    pad_id : int, default 0
        The id of the pad token in all dictionaries
    eos_id : int, default 2
        The id of the eos token
    Returns
    -------
    tuple
        (words_batch, chars_batch, labels_batch). words_batch is a torch.Tensor and has size (batch_size, max_sentence_length).
        chars_batch is a torch.Tensor and has size (batch_size * max_sentence_length, max_word_length).
    """

    sentences = [element[0] for element in batch] # sentences is a list of all list of words
    max_sentence_length = max(map(lambda x: len(x), sentences))
    max_word_length = max([max(map(lambda x: len(x), sentence)) for sentence in sentences])
    words_batch = []
    chars_batch = []
    labels_batch = []
    for words, labels in batch:
        words_indices = []
        chars_indices = []
        for word in words:
            word += [pad_id] * (max_word_length - len(word)) # id of the pad token must be 0
            words_indices += [word[0]]
            chars_indices += [word[1:]]
        words_indices += [pad_id] * (max_sentence_length - len(words))
        chars_indices += [[pad_id] * (max_word_length - 1)] * (max_sentence_length - len(words))

        for grammemes in labels:
            grammemes += [eos_id, pad_id] # id of the eos token must be 2

        words_batch += [words_indices]
        chars_batch += [chars_indices]
        labels_batch += [labels]

    words_batch = torch.tensor(words_batch, dtype=torch.int)
    chars_batch = torch.tensor(chars_batch, dtype=torch.int)
    chars_batch = chars_batch.view(-1, chars_batch.shape[2])
    return words_batch, chars_batch, labels_batch


if __name__ == "__main__":
    configurate()
    main(config)
