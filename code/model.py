"""Docstring for model.py."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vocab import Vocab
from dataset import CustomDataset
from encoder import Encoder
from decoder import Decoder
from config import configurate
from config import config
from sampler import BucketSampler


def main(conf):
    # print(conf)
    vocabulary = Vocab(conf)
    train_data = CustomDataset(conf, vocabulary, conf['train_files'],
                               sentences_pickle=conf['train_sentences_pickle'], training_set=True)
    valid_data = CustomDataset(conf, vocabulary, conf['valid_files'],
                               sentences_pickle=conf['valid_sentences_pickle'], training_set=False)
    test_data = CustomDataset(conf, vocabulary, conf['test_files'],
                              sentences_pickle=conf['test_sentences_pickle'], training_set=False)
    model = Model(conf, train_data).to(conf['device'])
    trainer = Trainer(conf, model, valid_data, test_data, subset_size=0).to(conf['device'])
    trainer.epoch_loops()


class Model(nn.Module):
    """Contains encoder and decoder, methods to calculate metrics and to convert decoder's output into strings.

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
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf['learning_rate'])

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

    def calculate_accuracy(self, predictions, targets):
        """Metrics is a ratio of correctly predicted tags to total number of tags.

        All grammemes in a tag must be predicted correctly for it to count as correct.
        """

        n_total, n_correct = 0, 0
        masked_predictions = masked_select(predictions.permute(1, 0),
                                           self.data.vocab.vocab["grammeme-index"][self.conf['EOS']])

        for tag, target in zip(masked_predictions, targets.permute(1, 0)):
            if target[0] != 0:
                n_total += 1
                n_correct += int(torch.equal(tag, target))

        return n_correct, n_total

    def predictions_to_grammemes(self, predictions):
        """Turns indices of predictions produced by decoder into actual grammemes (strings).

        Args:
            predictions (torch.Tensor): 2D Tensor containing indices.

        Returns:
        list: List of lists of predicions.
        """

        tags = []
        for tag_indices in predictions:
            tag = []
            for grammeme_index in tag_indices:
                tag += [self.data.vocab.vocab['index-grammeme'][grammeme_index.item()]]
            tags += [tag]
        return tags


class Trainer(nn.Module):
    """Class performs training. More info will be added later.

    Args:
        conf (dict): Dictionary with configuration parameters.
        model (Model): Instance of class containing model parameters.
        valid_data (CustomDataset): Dataset for validation.
        test_data (CustomDataset): Dataset for testing.
        subset_size (float or int): The part of dataset from model.data that will be used.
            If int, treated as the number of samples from model.data with 0 treated as a whole dataset.
            If float, should be between 0 and 1, treated as the proportion of the dataset used during training.
    """

    def __init__(self, conf, model, valid_data, test_data, subset_size=0):
        super().__init__()
        self.conf = conf
        self.model = model

        if subset_size == 0:
            train_subset = self.model.data
            valid_subset = valid_data
            test_subset = test_data
        elif isinstance(subset_size, int) and subset_size > 0:
            train_subset = subset_from_dataset(self.model.data, subset_size)
            valid_subset = subset_from_dataset(valid_data, subset_size)
            test_subset = subset_from_dataset(test_data, subset_size)
        elif isinstance(subset_size, float) and 0.0 < subset_size < 1.0:
            train_subset = subset_from_dataset(self.model.data, int(subset_size * len(self.model.data)))
            valid_subset = subset_from_dataset(valid_data, int(subset_size * len(valid_data)))
            test_subset = subset_from_dataset(test_data, int(subset_size * len(test_data)))
        else:
            raise TypeError("Only non-negative ints and floats between 0 and 1 are allowed")
        if self.conf['bucket_train_data']:
            sampler = BucketSampler(train_subset, self.conf['sentence_train_batch_size'])
        else:
            sampler = None
        self.train_loader = DataLoader(train_subset, batch_size=self.conf['sentence_train_batch_size'],
                                       collate_fn=collate_batch, sampler=sampler)
        self.valid_loader = DataLoader(valid_subset, batch_size=self.conf['sentence_eval_batch_size'],
                                       collate_fn=collate_batch)
        self.test_loader = DataLoader(test_subset, batch_size=self.conf['sentence_eval_batch_size'],
                                      collate_fn=collate_batch)

        self.writer = SummaryWriter()
        self.current_epoch = 0
        self.print_every = 20  # write to log every so many iteration
        self.best_metrics = torch.inf
        self.no_improv = 0

    def epoch_loops(self):
        print(f"{len(self.train_loader)} batches in train")
        print(f"{len(self.valid_loader)} batches in valid")
        print(f"{len(self.test_loader)} batches in test")

        for epoch in range(self.conf['max_epochs']):
            self.train_epoch()
            valid_accuracy, valid_loss = self.valid_epoch()
            print(f"valid accuracy at epoch {self.current_epoch}: {valid_accuracy}")
            print(f"valid loss: {valid_loss}")
            if valid_loss >= self.best_metrics:
                self.no_improv += 1
                if self.no_improv >= self.conf['no_improv']:
                    print(f"No improvement for {self.conf['no_improv']} epochs, stopping early")
                    break
            else:
                self.no_improv = 0
                self.best_metrics = valid_loss
                torch.save(self.model, config['model'])  # put the path elsewhere and create folder if necessary
            self.current_epoch += 1

    def train_epoch(self):
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        running_loss = 0.0
        # current_epoch_tags = [] # stores tags for this epoch

        for iteration, (words_batch, chars_batch, labels_batch) in progress_bar:
            self.model.optimizer.zero_grad()
            _, probabilities = self.model(words_batch, chars_batch, labels_batch)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token

            loss = self.model.loss(probabilities.permute(1, 2, 0), targets.permute(1, 0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf['clip'])
            self.model.optimizer.step()
            running_loss += loss.item()

            if (self.current_epoch * len(self.train_loader) + iteration) % self.print_every == 0:
                self.writer.add_scalar("training loss",
                                       running_loss / self.print_every,
                                       self.current_epoch * len(self.train_loader) + iteration)
                running_loss = 0.0

        # print("One train epoch complete")

    def valid_epoch(self):
        self.model.eval()

        # code similar to train_epoch
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        running_error = 0.0
        correct, total = 0, 0
        for iteration, (words_batch, chars_batch, labels_batch) in progress_bar:
            predictions, probabilities = self.model(words_batch, chars_batch, labels_batch)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token

            error = self.model.loss(probabilities.permute(1, 2, 0), targets.permute(1, 0))
            running_error += error.item()

            correct_batch, total_batch = self.model.calculate_accuracy(predictions, targets)
            correct += correct_batch
            total += total_batch
        valid_accuracy = correct / total
        valid_loss = running_error / (self.current_epoch + 1) * len(self.valid_loader)
        self.writer.add_scalar("valid accuracy",
                               valid_accuracy,
                               (self.current_epoch + 1) * len(self.valid_loader))
        self.writer.add_scalar("valid loss",
                               valid_loss,
                               (self.current_epoch + 1) * len(self.valid_loader))
        return valid_accuracy, valid_loss

    def test_epoch(self):
        self.model.eval()

        # code similar to train_epoch
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        correct, total = 0, 0
        for iteration, (words_batch, chars_batch, labels_batch) in progress_bar:
            predictions, probabilities = self.model(words_batch, chars_batch, labels_batch)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token

            correct_batch, total_batch = self.model.calculate_accuracy(predictions, targets)
            correct += correct_batch
            total += total_batch
        test_accuracy = correct / total
        self.writer.add_scalar("test accuracy",
                               test_accuracy,
                               (self.current_epoch + 1) * len(self.valid_loader))
        return test_accuracy


def collate_batch(batch, pad_id=0, sos_id=1, eos_id=2):  # do all preprocessing here
    """Takes batch created with CustomDataset and performs padding.

    batch consists of indices of words, chars, and labels (grammemes). The returned batches are tensors.

    Args:
        batch (list): Single batch to be collated.
            Batch consists of tuples (words, labels) generated by CustomDataset.
        pad_id (int, default 0): The id of the pad token in all dictionaries.
        sos_id (int, default 1): The id of the sos (start of sequence) token.
        eos_id (int, default 2): The id of the eos (end of sequence) token.

    Returns:
        tuple: (words_batch, chars_batch, labels_batch).
            All of them have type torch.Tensor. Size of words_batch is (max_sentence_length, batch_size).
            Size of chars_batch is (batch_size * max_sentence_length, max_word_length). Size of labels_batch is
            (max_label_length, batch_size * max_sentence_length)
    """

    sentences = [element[0] for element in batch]  # sentences is a list of all list of words
    tags = [element[1] for element in batch]  # tags is a list of all lists of grammemes
    max_sentence_length = max(map(lambda x: len(x), sentences))
    max_word_length = max([max(map(lambda x: len(x), sentence)) for sentence in sentences])
    max_label_length = 2 + max([max(map(lambda x: len(x), tag)) for tag in tags])  # +2 because of the sos and eos token
    words_batch = []
    chars_batch = []
    labels_batch = []
    for words, labels in batch:
        words_indices = []
        chars_indices = []
        labels_indices = []
        for word in words:
            word += [pad_id] * (max_word_length - len(word))  # id of the pad token must be 0
            words_indices += [word[0]]
            chars_indices += [word[1:]]
        words_indices += [pad_id] * (max_sentence_length - len(words))
        chars_indices += [[pad_id] * (max_word_length - 1)] * (max_sentence_length - len(words))

        for label in labels:
            label.insert(0, sos_id)  # id of the sos token must be 1
            label += [eos_id]  # id of the eos token must be 2
            label += [pad_id] * (max_label_length - len(label))
            labels_indices += [label]
        labels_indices += [[pad_id] * max_label_length] * (max_sentence_length - len(words))

        words_batch += [words_indices]
        chars_batch += [chars_indices]
        labels_batch += [labels_indices]

    words_batch = torch.tensor(words_batch, dtype=torch.long)
    words_batch = words_batch.transpose(1, 0)
    chars_batch = torch.tensor(chars_batch, dtype=torch.long)
    chars_batch = chars_batch.view(-1, chars_batch.shape[2])
    labels_batch = torch.tensor(labels_batch, dtype=torch.long)
    labels_batch = labels_batch.view(-1, labels_batch.shape[2]).permute(1, 0)
    return words_batch.to(config['device']), chars_batch.to(config['device']), labels_batch.to(config['device'])


def subset_from_dataset(data, n):
    """Outputs first n entries from data (type Dataset) as another dataset."""
    return Subset(data, range(n))


def masked_select(a, value):
    """Zero all elements that come after a given value in a row. Used for zeroing elements after EOS token.

    Args:
        a (torch.Tensor): Input tensor.
        value (a.dtype): Value after which all elements should be equal to zero (EOS token index).

    Returns:
        torch.Tensor: Masked tensor of the same shape as a.

    Examples:
        >>> input_tensor = torch.Tensor([[1., 2., 3., 4., 99., 5., 2., 1.],
                              [1., 99., 99., 4., 3., 5., 99., 3.],
                              [1., 3., 3., 4., 1., 5., 2., 1.]])
        >>> print(masked_select(input_tensor, 99))
        tensor([[ 1.,  2.,  3.,  4., 99.,  0.,  0.,  0.],
                [ 1., 99.,  0.,  0.,  0.,  0.,  0.,  0.],
                [ 1.,  3.,  3.,  4.,  1.,  5.,  2.,  1.]])
    """

    mask = []
    rng = torch.arange(0, a.shape[1]).to(config['device'])
    for row in a:
        equal = torch.isin(row, value)
        if equal.nonzero().shape[0] != 0:
            mask.append(torch.le(rng, equal.nonzero()[0]))
        else:
            mask.append(rng)

    mask = torch.stack(mask)
    return a * mask


if __name__ == "__main__":
    configurate()
    main(config)
