"""
endoder.py has class Encoder that creates batches of words embeddings
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
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
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(data.embeddings)).float() # from_pretrained only outputs torch.float64
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
        output, _ = self.wordLSTM(words)
        # final shape is (batch_size, max_sentence_length, 2 * word_LSTM_hidden)
        # print(output.shape)
        return output
