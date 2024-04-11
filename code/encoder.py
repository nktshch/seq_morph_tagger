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
                                hidden_size=self.conf['char_LSTM_hidden'], bidirectional=self.conf['char_LSTM_bidirectional'], batch_first=True)
        self.wordLSTMcell = nn.LSTMCell(input_size=(self.conf['word_embeddings_dimension'] + self.conf['char_LSTM_hidden'] * self.conf['char_LSTM_directions']),
                                        hidden_size=self.conf['word_LSTM_hidden'])
        self.wordDropout_input = nn.Dropout(p=self.conf['word_LSTM_input_dropout'])
        self.wordDropout_state = nn.Dropout(p=self.conf['word_LSTM_state_dropout'])
        self.wordDropout_output = nn.Dropout(p=self.conf['word_LSTM_output_dropout'])

    def forward(self, words_batch, chars_batch):
        """
        Takes batches of indices of words and chars and creates embeddings with LSTM.
        PyTorch LSTM module doesn't return cell states by default. That is why we have to use LSTMCell in a loop.

        Parameters
        ----------
        words_batch : torch.Tensor
            Tensor of words indices for every word in a batch. Size (max_sentence_length, batch_size)
        chars_batch : torch.Tensor
            Tensor of chars indices for every word in a batch. Size (batch_size * max_sentence_length, max_word_length)
        Returns
        -------
        tuple
            Tuple consists of two tensors - one with hidden states, and one with cell states of the word LSTM.
            The shape of each tensor is (max_sentence_length, batch_size, grammeme_LSTM_hidden)
        """

        current_batch_size = words_batch.shape[1]
        words = self.word_embeddings(words_batch)
        chars = self.char_embeddings(chars_batch)
        # words has shape (max_sentence_length, batch_size, word_embeddings_dimension)
        # chars has shape (batch_size * max_sentence_length, max_word_length, char_embeddings_dimension)
        _, (hn, cn) = self.charLSTM(chars)
        # hn has shape (char_LSTM_directions, batch_size * max_sentence_length, char_LSTM_hidden)
        chars = hn.permute(1, 0, 2).contiguous().view(current_batch_size, -1, hn.shape[2] * self.conf['char_LSTM_directions']).permute(1, 0, 2)
        # we have to transpose twice because of how .view() changes shapes of tensors. Thoughtless usage can lead to serious mistakes!
        # chars has shape (max_sentence_length, batch_size, char_LSTM_directions * char_LSTM_hidden)
        words = torch.concat((words, chars), dim=2)
        words = self.wordDropout_input(words)
        # words has shape (max_sentence_length, batch_size, word_embeddings_dimension + char_LSTM_directions * char_LSTM_hidden)

        hidden_forward, cell_forward = self.loop(words)

        if self.conf['word_LSTM_bidirectional']:
            hidden_backward, cell_backward = self.loop(words.flip(dims=[0]))

            hidden_backward = hidden_backward.flip(dims=[0])
            cell_backward = cell_backward.flip(dims=[0])

            hidden = torch.concat((hidden_forward, hidden_backward), dim=2)
            cell = torch.concat((cell_forward, cell_backward), dim=2)
            # final shape is (max_sentence_length, batch_size, grammeme_LSTM_hidden)

            hidden = self.wordDropout_output(hidden)
            cell = self.wordDropout_output(cell)
            return hidden, cell

        hidden = hidden_forward
        cell = cell_forward
        # final shape is (max_sentence_length, batch_size, grammeme_LSTM_hidden)

        hidden = self.wordDropout_output(hidden)
        cell = self.wordDropout_output(cell)
        return hidden, cell

    def loop(self, words):
        hk = torch.zeros((words.size(dim=1), self.wordLSTMcell.hidden_size)).to(self.conf['device'])
        ck = torch.zeros((words.size(dim=1), self.wordLSTMcell.hidden_size)).to(self.conf['device'])
        hidden = []
        cell = []
        for word in words:
            hk = self.wordDropout_state(hk)
            ck = self.wordDropout_state(ck)
            hk, ck = self.wordLSTMcell(word, (hk, ck))
            hidden += [hk]
            cell += [ck]
        hidden = torch.stack(hidden)
        cell = torch.stack(cell)

        return hidden, cell
