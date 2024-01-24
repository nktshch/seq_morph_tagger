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
        self.grammemeLSTMcell = nn.LSTMCell(input_size=self.conf['grammeme_embeddings_dimension'], hidden_size=self.conf['grammeme_LSTM_hidden'])
        self.grammemeDropout = nn.Dropout(p=self.conf['grammeme_LSTM_input_dropout'])
        self.linear = nn.Linear(in_features=self.conf['grammeme_LSTM_hidden'], out_features=len(data.vocab.vocab['grammeme-index']))

    def forward(self, labels_batch, decoder_hidden, decoder_cell):
        # during training, these will be fed one at a time, instead of outputs at each time step
        labels = self.grammeme_embeddings(labels_batch) # embeddings of grammemes,
        # size (max_grammeme_length, batch_size * max_sentence_length, grammeme_embeddings_dimension)
        # size[0] = loop length (sequence length), size[1] = size of the batch, size[3] = input size
        hk = decoder_hidden
        ck = decoder_cell
        predictions = []
        probabilities = []
        # this is incomplete!
        if self.training:
            for grammemes in labels:
                hk, ck = self.grammemeLSTMcell(grammemes, (hk, ck))
                probabilities_batch = self.linear(hk)
                probabilities += [probabilities_batch]
                predictions_batch = torch.argmax(probabilities_batch, dim=1)
                predictions += [predictions_batch]

        else: # using generated grammemes as the next input
            grammemes = labels[0]
            for _ in range(self.conf['decoder_max_iterations']):
                hk, ck = self.grammemeLSTMcell(grammemes, (hk, ck))
                probabilities_batch = self.linear(hk)
                probabilities += [probabilities_batch]
                predictions_batch = torch.argmax(probabilities_batch, dim=1)
                grammemes = self.grammeme_embeddings(predictions_batch)
                predictions += [predictions_batch]
        probabilities = torch.stack(probabilities)
        predictions = torch.stack(predictions).permute(1, 0)
        # predictions has size (batch_size * max_sentence_length, max_grammeme_length)
        return predictions, probabilities
