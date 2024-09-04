"""Docstring for trainer.py"""

from sampler import BucketSampler
from utils import collate_batch, subset_from_dataset, calculate_accuracy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
                                       collate_fn=lambda x: tuple(x_.to(self.conf['device']) for x_ in collate_batch(x)),
                                       sampler=sampler)
        self.valid_loader = DataLoader(valid_subset, batch_size=self.conf['sentence_eval_batch_size'],
                                       collate_fn=lambda x: tuple(x_.to(self.conf['device']) for x_ in collate_batch(x)))
        self.test_loader = DataLoader(test_subset, batch_size=self.conf['sentence_eval_batch_size'],
                                      collate_fn=lambda x: tuple(x_.to(self.conf['device']) for x_ in collate_batch(x)))

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
                torch.save(self.model, self.conf['model'])  # put the path elsewhere and create folder if necessary
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
