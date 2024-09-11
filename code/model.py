"""Docstring for model.py."""

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Model(nn.Module):
    """Contains encoder and decoder.

    Args:
        conf (dict): Dictionary with configuration parameters.
        data (CustomDataset): Instance of class containing dataset.
    """

    def __init__(self, conf, data):
        super().__init__()
        self.conf = conf
        self.data = data
        self.encoder = Encoder(self.conf, self.data)  # provides words embeddings
        self.decoder = Decoder(self.conf, self.data)
        self.grammeme_embeddings = None

    def forward(self, words_batch, chars_batch, labels_batch):
        """Uses Encoder and Decoder to perform one pass on a sinle batch.

        Args:
            words_batch (torch.Tensor): Tensor of words indices for every word in a batch.
                Size (max_sentence_length, batch_size).
            chars_batch (torch.Tensor): Tensor of chars indices for every word in a batch.
                Size (batch_size * max_sentence_length, max_word_length).
            labels_batch (torch.Tensor): Tensor of labels indices for every word in a batch.
                Size (max_label_length, batch_size * max_sentence_length).

        Returns:
            tuple: Tuple consists of predicted grammemes and their probabilities.
        """

        # shape (max_sentence_length, batch_size, grammeme_LSTM_hidden)
        encoder_hidden, encoder_cell = self.encoder(words_batch, chars_batch)
        decoder_hidden = encoder_hidden.permute(1, 0, 2).reshape(-1, encoder_hidden.size(dim=2))
        decoder_cell = encoder_cell.permute(1, 0, 2).reshape(-1, encoder_cell.size(dim=2))
        predictions, probabilities = self.decoder(labels_batch, decoder_hidden, decoder_cell)

        return predictions, probabilities


# if __name__ == "__main__":
#     configurate()
#     main(config)
