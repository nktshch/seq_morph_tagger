"""Contains class Model that has encoder and decoder and produces predictions and probabilities of grammemes."""

import torch
import torch.nn as nn

from model.layers.encoder import Encoder
from model.layers.decoder import Decoder


class Model(nn.Module):
    """Contains encoder and decoder.

    Args:
        conf (dict): Dictionary with configuration parameters.
        vocab (Vocab): Instance of class containing vocab.

    Attributes:
        encoder: Encoder class from encoder.py
        decoder: Decoder class from decoder.py
    """

    def __init__(self, conf, vocab):
        super().__init__()
        self.conf = conf
        self.vocab = vocab

        self.encoder = Encoder(self.conf, self.vocab)
        self.decoder = Decoder(self.conf, self.vocab)

    def forward(self, words_batch, chars_batch, labels_batch=None):
        """Uses Encoder and Decoder to perform one pass on a sinle batch.

        Args:
            words_batch (torch.Tensor): Tensor of words indices for every word in a batch.
                Size (max_sentence_length, batch_size).
            chars_batch (torch.Tensor): Tensor of chars indices for every word in a batch.
                Size (batch_size * max_sentence_length, max_word_length).
            labels_batch (torch.Tensor, default None): Tensor of labels indices for every word in a batch.
                Size (max_label_length, batch_size * max_sentence_length).
                If None, decoder will use generated grammemes for the next prediction (inference mode).
                Otherwise, decoder will use labels_batch (training mode)

        Returns:
            tuple: Tuple consists of predicted grammemes and their probabilities.
        """

        # shape (max_sentence_length, batch_size, grammeme_LSTM_hidden)
        encoder_hidden, encoder_cell = self.encoder(words_batch, chars_batch)
        decoder_hidden = encoder_hidden.permute(1, 0, 2).reshape(-1, encoder_hidden.size(dim=2))
        decoder_cell = encoder_cell.permute(1, 0, 2).reshape(-1, encoder_cell.size(dim=2))
        predictions, probabilities = self.decoder(decoder_hidden, decoder_cell, labels_batch)

        return predictions, probabilities
