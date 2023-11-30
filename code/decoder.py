"""
Docstring for decoder.py
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Docstring for Decoder
    """

    def __init__(self, conf, data):
        super().__init__()
        self.conf = conf
        self.data = data
        self.grammeme_embeddings = nn.Embedding(len(data.vocab.vocab['grammeme-index']), self.conf["grammeme_embeddings_dimension"])
        self.grammemeLSTMcell = nn.LSTMCell(input_size=self.conf['grammeme_embeddings_dimension'], hidden_size=2 * self.conf['word_LSTM_hidden'])
        self.grammemeDropout = nn.Dropout(p=self.conf['grammeme_LSTM_input_dropout'])

    def forward(self, labels_batch, decoder_hidden):
        start_indices = torch.zeros(decoder_hidden.shape[0]).fill_(self.data.vocab.vocab['grammeme-index']["$SOS$"]).int()
        train_indices = torch.concat((start_indices.view(1, -1), labels_batch[:, :-1].transpose(0, 1)), dim=0)
        # during training, these will be fed one at a time, instead of outputs at each time step

        train_grammemes = self.grammeme_embeddings(train_indices) # embeddings of grammemes,
        # size (max_grammeme_length, batch_size * max_sentence_length, grammeme_embeddings_dimension)
        # size[0] = loop length (sequence length), size[1] = size of the batch, size[3] = input size
        hk = decoder_hidden
        ck = torch.zeros(hk.size()) # we should use real c0, produced by Encoder
        hidden = []
        for grammemes in train_grammemes:
            hk1, ck1 = self.grammemeLSTMcell(grammemes, (hk, ck))
            hk = hk1
            ck = ck1
            hidden += [hk1]

        # hidden has size (max_grammeme_length, batch_size * max_sentence_length, 2 * word_LSTM_hidden)
        return hidden
