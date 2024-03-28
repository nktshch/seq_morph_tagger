"""
Docstring for model.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from vocab import Vocab
from dataset import CustomDataset
from encoder import Encoder
from decoder import Decoder
from config import configurate
from config import config


def main(conf):
    # print(conf)
    vocabulary = Vocab(conf)
    train_data = CustomDataset(conf, vocabulary, conf['train_files'],
                               sentences_pickle=conf['train_sentences_pickle'], training_set=True)
    valid_data = CustomDataset(conf, vocabulary, conf['valid_files'],
                               sentences_pickle=conf['valid_sentences_pickle'], training_set=False)
    model = Model(conf, train_data).to(conf['device'])
    trainer = Trainer(conf, model, valid_data, subset_size=100).to(conf['device'])
    trainer.epoch_loops()
    # train_loader = torch.utils.data.DataLoader(model.data, batch_size=conf['sentence_batch_size'], collate_fn=collate_batch)
    # progress_bar = tqdm(enumerate(train_loader), disable=True)
    # for _, (words_batch, chars_batch, labels_batch) in progress_bar:
    #     tags = model(words_batch, chars_batch, labels_batch)


class Model(nn.Module):
    """
    Class takes batches produced by DataLoader and assignes embeddings to words using WordEmbeddings class.
    Creates an instance of CustomDataset.

    Parameters
    ----------
    conf : dict
        Dictionary with configuration parameters
    data : CustomDataset
        Instance of class containing dataset
    """

    def __init__(self, conf, data):
        super().__init__()
        self.conf = conf
        self.data = data
        self.encoder = Encoder(self.conf, self.data) # provides words embeddings
        self.decoder = Decoder(self.conf, self.data)
        self.grammeme_embeddings = None

    def forward(self, words_batch, chars_batch, labels_batch):
        """
        Uses Encoder and Decoder to perform one pass on a sinle batch.

        Parameters
        ----------
        words_batch : torch.Tensor
            Tensor of words indices for every word in a batch. Size (max_sentence_length, batch_size)
        chars_batch : torch.Tensor
            Tensor of chars indices for every word in a batch. Size (batch_size * max_sentence_length, max_word_length)
        labels_batch : torch.Tensor
            Tensor of labels indices for every word in a batch. Size (max_label_length, batch_size * max_sentence_length)
        Returns
        -------
        tuple
            Tuple consists of predicted grammemes and their probabilities.
        """

        encoder_hidden, encoder_cell = self.encoder(words_batch, chars_batch) # shape (max_sentence_length, batch_size, grammeme_LSTM_hidden)
        decoder_hidden = encoder_hidden.view(-1, encoder_hidden.size(dim=2))
        decoder_cell = encoder_cell.view(-1, encoder_cell.size(dim=2))
        predictions, probabilities = self.decoder(labels_batch, decoder_hidden, decoder_cell)

        return predictions, probabilities

    def predictions_to_grammemes(self, predictions):
        """
        Turns indices of predictions produced by decoder into actual grammemes (strings)

        Parameters
        ----------
        predictions : torch.Tensor
            2D Tensor containing indices
        Returns
        -------
        list
            List of lists of predicions
        """
        tags = []
        for tag_indices in predictions:
            tag = []
            for grammeme_index in tag_indices:
                tag += [self.data.vocab.vocab['index-grammeme'][grammeme_index.item()]]
            tags += [tag]
        return tags


class Trainer(nn.Module):
    def __init__(self, conf, model, valid_data, subset_size=0):
        """
        Class performs training. More info will be added later

        Parameters
        ----------
        conf : dict
            Dictionary with configuration parameters
        model : Model
            Instance of class containing model parameters
        valid_data : CustomDataset
            Dataset for validation
        subset_size : float of int
            Whether to use full dataset from model.data, or only some part of it. If int, treated as the number
            of samples from model.data. If float, should be between 0 and 1, treated as the proportion of the dataset used
            during training. If 0, whole dataset is used
        """
        super().__init__()
        self.conf = conf
        self.model = model

        if subset_size == 0:
            train_subset = self.model.data
            valid_subset = valid_data
        elif isinstance(subset_size, int) and subset_size > 0:
            train_subset = subset_from_dataset(self.model.data, subset_size)
            valid_subset = subset_from_dataset(valid_data, subset_size)
        elif isinstance(subset_size, float) and 0.0 < subset_size < 1.0:
            train_subset = subset_from_dataset(self.model.data, int(subset_size * len(self.model.data)))
            valid_subset = subset_from_dataset(valid_data, int(subset_size * len(valid_data)))
        else:
            raise TypeError("Only positive ints and floats between 0 and 1 are allowed")
        self.train_loader = DataLoader(train_subset, batch_size=self.conf['sentence_batch_size'], collate_fn=collate_batch)
        self.valid_loader = DataLoader(valid_subset, batch_size=self.conf['sentence_batch_size'], collate_fn=collate_batch)

        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.conf['learning_rate'])
        self.writer = SummaryWriter()
        self.current_epoch = 0

    def epoch_loops(self):
        print(f"{len(self.train_loader)} batches")
        # for epoch in range(self.conf['max_epochs']):
        for epoch in range(2):
            self.train_epoch()
            metrics, error = self.evaluate()
            self.current_epoch += 1


    def train_epoch(self):
        self.model.encoder.train()
        self.model.decoder.train()
        progress_bar = enumerate(self.train_loader)
        running_loss = 0.0
        print_every = 20
        current_epoch_tags = [] # stores tags for this epoch

        for iteration, (words_batch, chars_batch, labels_batch) in progress_bar:
            self.optimizer.zero_grad()
            _, probabilities = self.model(words_batch, chars_batch, labels_batch)

            targets = labels_batch[1:].to(torch.long) # slice is taken to ignore SOS token
            print(targets.shape, probabilities.shape)
            loss = self.loss(probabilities.permute(1, 2, 0), targets.permute(1, 0))
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (self.current_epoch * len(self.train_loader) + iteration) % print_every == 0:
                self.writer.add_scalar("training loss", running_loss / print_every, self.current_epoch * len(self.train_loader) + iteration)
                running_loss = 0.0

        print("One train epoch complete")


    def evaluate(self):
        self.model.encoder.eval()
        self.model.decoder.eval()
        metrics = 0
        error = 0

        # code similar to train_epoch
        progress_bar = enumerate(self.valid_loader)
        for iteration, (words_batch, chars_batch, labels_batch) in progress_bar:
            tags, _ = self.model(words_batch, chars_batch, labels_batch)

            targets = labels_batch[1:].to(torch.long)
            print(targets.shape, tags.shape)
            # calculate error and metrics

        print("One valid epoch complete")
        return metrics, error


def collate_batch(batch, pad_id=0, sos_id=1, eos_id=2): # do all preprocessing here
    """
    Function takes batch created with CustomDataset and performs padding. batch consists of indices of words,
    chars, and labels (grammemes). The returned batches are tensors.

    Parameters
    ---------
    batch : list
        A single batch to be collated. Batch consists of tuples (words, labels) generated by CustomDataset
    pad_id : int, default 0
        The id of the pad token in all dictionaries
    sos_id : int, default 1
        The id of the sos (start of sequence) token
    eos_id : int, default 2
        The id of the eos (end of sequence) token
    Returns
    -------
    tuple
        (words_batch, chars_batch, labels_batch). All of them have type torch.Tensor. Size of words_batch is (max_sentence_length, batch_size).
        Size of chars_batch is (batch_size * max_sentence_length, max_word_length). Size of labels_batch is
        (max_label_length, batch_size * max_sentence_length)
    """

    sentences = [element[0] for element in batch] # sentences is a list of all list of words
    tags = [element[1] for element in batch] # tags is a list of all lists of grammemes
    max_sentence_length = max(map(lambda x: len(x), sentences))
    max_word_length = max([max(map(lambda x: len(x), sentence)) for sentence in sentences])
    max_label_length = 2 + max([max(map(lambda x: len(x), tag)) for tag in tags]) # +2 because of the eos token
    words_batch = []
    chars_batch = []
    labels_batch = []
    for words, labels in batch:
        words_indices = []
        chars_indices = []
        labels_indices = []
        for word in words:
            word += [pad_id] * (max_word_length - len(word)) # id of the pad token must be 0
            words_indices += [word[0]]
            chars_indices += [word[1:]]
        words_indices += [pad_id] * (max_sentence_length - len(words))
        chars_indices += [[pad_id] * (max_word_length - 1)] * (max_sentence_length - len(words))

        for label in labels:
            label.insert(0, sos_id) # id of the sos token must be 1
            label += [eos_id] # id of the eos token must be 2
            label += [pad_id] * (max_label_length - len(label))
            labels_indices += [label]
        labels_indices += [[pad_id] * max_label_length] * (max_sentence_length - len(words))

        words_batch += [words_indices]
        chars_batch += [chars_indices]
        labels_batch += [labels_indices]

    words_batch = torch.tensor(words_batch, dtype=torch.int)
    words_batch = words_batch.transpose(1, 0)
    chars_batch = torch.tensor(chars_batch, dtype=torch.int)
    chars_batch = chars_batch.view(-1, chars_batch.shape[2])
    labels_batch = torch.tensor(labels_batch, dtype=torch.int)
    labels_batch = labels_batch.view(-1, labels_batch.shape[2]).permute(1, 0)
    return words_batch.to(config['device']), chars_batch.to(config['device']), labels_batch.to(config['device'])

def subset_from_dataset(data, n):
    """
    Outputs first n entries from data (type Dataset) as another dataset
    """
    return Subset(data, range(n))


if __name__ == "__main__":
    configurate()
    main(config)
