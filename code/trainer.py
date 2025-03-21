"""Docstring for trainer.py"""

from data_preparation.sampler import BucketSampler

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment as lsa
from tqdm import tqdm
import fasttext

class Trainer(nn.Module):
    """Class performs training. More info will be added later.

    Args:
        conf (dict): Dictionary with configuration parameters.
        model (Model): Instance of class containing model parameters.
        valid_data (CustomDataset): Dataset for validation.
        test_data (CustomDataset): Dataset for testing.
        run_number: Run / random seed number.
        subset_size (float or int): The part of dataset from model.data that will be used.
            If int, treated as the number of samples from model.data with 0 treated as a whole dataset.
            If float, should be between 0 and 1, treated as the proportion of the dataset used during training.
    """

    def __init__(self, conf, model, train_data, valid_data, test_data, run_number=0, subset_size=0):
        super().__init__()
        self.conf = conf
        self.model = model
        self.vocab = model.vocab
        self.label_length = len(self.vocab.categories_by_freq) if self.conf['same_length'] is True \
            else None
        self.directory = f"{self.conf['model']}/seed_{run_number}"

        if subset_size == 0:
            train_subset = train_data
            valid_subset = valid_data
            test_subset = test_data
        elif isinstance(subset_size, int) and subset_size > 0:
            train_subset = subset_from_dataset(train_data, subset_size)
            valid_subset = subset_from_dataset(valid_data, subset_size)
            test_subset = subset_from_dataset(test_data, subset_size)
        elif isinstance(subset_size, float) and 0.0 < subset_size < 1.0:
            train_subset = subset_from_dataset(train_data, int(subset_size * len(train_data)))
            valid_subset = subset_from_dataset(valid_data, int(subset_size * len(valid_data)))
            test_subset = subset_from_dataset(test_data, int(subset_size * len(test_data)))
        else:
            raise TypeError("Only non-negative ints and floats between 0 and 1 are allowed")
        if self.conf['bucket_train_data']:
            sampler = BucketSampler(train_subset, self.conf['sentence_train_batch_size'])
        else:
            sampler = None
        self.train_loader = DataLoader(train_subset, batch_size=self.conf['sentence_train_batch_size'],
                                       collate_fn=lambda item: collate_batch(item, label_length=self.label_length),
                                       sampler=sampler)
        self.valid_loader = DataLoader(valid_subset, batch_size=self.conf['sentence_eval_batch_size'],
                                       collate_fn=lambda item: collate_batch(item, label_length=self.label_length)) if valid_subset else []
        self.test_loader = DataLoader(test_subset, batch_size=self.conf['sentence_eval_batch_size'],
                                      collate_fn=lambda item: collate_batch(item, label_length=self.label_length)) if test_subset else []

        self.xe_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.nll_loss = nn.NLLLoss(reduction='mean', ignore_index=0)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf['learning_rate'])

        self.writer = SummaryWriter(log_dir=self.directory)
        self.current_epoch = 0
        self.print_every = 20  # write to log every so many iteration
        self.best_accuracy = 0
        self.no_improv = 0

        print(f"{len(self.train_loader)} batches in train")
        print(f"{len(self.valid_loader)} batches in valid")
        print(f"{len(self.test_loader)} batches in test")


    def epoch_loops(self, oov_pretrained_vocab):
        for epoch in range(self.conf['max_epochs']):
            self.model.train()
            self.train_epoch()

            if len(self.valid_loader) > 0:
                self.model.eval()
                with torch.no_grad():
                    valid_accuracy, valid_loss = self.valid_epoch(oov_pretrained_vocab)
                if valid_accuracy <= self.best_accuracy:
                    self.no_improv += 1
                    if self.no_improv >= self.conf['no_improv']:
                        print(f"No improvement in accuracy for {self.conf['no_improv']} epochs, stopping early")
                        break
                else:
                    self.no_improv = 0
                    self.best_accuracy = valid_accuracy
                    torch.save([self.conf, self.model.state_dict()],
                               f"{self.directory}/model.pt")
                self.current_epoch += 1


    def train_epoch(self):
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        # current_epoch_tags = [] # stores tags for this epoch

        for iteration, (words_batch, chars_batch, labels_batch, _) in progress_bar:
            words_batch = words_batch.to(self.conf['device'])
            chars_batch = chars_batch.to(self.conf['device'])
            labels_batch = labels_batch.to(self.conf['device'])

            self.optimizer.zero_grad()
            _, probabilities = self.model(words_batch, chars_batch, labels_batch)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token
            probabilities = probabilities[:len(targets)]
            # probabilities has shape (max_label_length, max_sentence_length * batch_size, grammemes_in_vocab)
            # targets has shape (max_label_length, max_sentence_length * batch_size)

            targets = targets.permute(1, 0)
            probabilities = probabilities.permute(1, 2, 0)

            if self.conf['loss'] == "xe":
                loss = self.xe_loss(probabilities, targets)
            elif self.conf['loss'] == "oaxe":
                loss, _ = self.oaxe_loss(probabilities, targets)
            else:
                raise ValueError(f"Unknown loss: {self.conf['loss']}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf['clip'])
            self.optimizer.step()

            if iteration % self.print_every == 0:
                self.writer.add_scalar("training loss",
                                       loss.item(),
                                       self.current_epoch * len(self.train_loader) + iteration)

        # print("One train epoch complete")


    def valid_epoch(self, oov_pretrained_vocab):
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), colour='#bbbbff')
        running_error = 0.0
        correct, total = 0, 0
        for iteration, (words_batch, chars_batch, labels_batch, raw_sentences) in progress_bar:
            words_batch = words_batch.to(self.conf['device'])
            chars_batch = chars_batch.to(self.conf['device'])
            labels_batch = labels_batch.to(self.conf['device'])

            fasttext_embeddings = []
            mask_embeddings = torch.zeros(words_batch.shape, dtype=bool, device=words_batch.device)
            for i, ith_words in enumerate(words_batch):
                for j, word_index in enumerate(ith_words):
                    if word_index.item() == 1:
                        if raw_sentences[j][i].lower() in oov_pretrained_vocab.keys():
                            fasttext_embeddings += [oov_pretrained_vocab[raw_sentences[j][i].lower()]]
                            mask_embeddings[i, j] = True

            oov = None
            if torch.any(mask_embeddings):
                fasttext_embeddings = torch.tensor(np.array(fasttext_embeddings)).to(self.conf['device'])
                assert mask_embeddings.sum().item() == fasttext_embeddings.shape[0]
                oov = (fasttext_embeddings, mask_embeddings)

            predictions, probabilities = self.model(words_batch, chars_batch, None, oov=oov)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token
            probabilities = probabilities[:len(targets)]
            # probabilities has shape (max_label_length, max_sentence_length * batch_size, grammemes_in_vocab)
            # targets has shape (max_label_length, max_sentence_length * batch_size)

            targets = targets.permute(1, 0)
            predictions = predictions.permute(1, 0)
            probabilities = probabilities.permute(1, 2, 0)

            if self.conf['loss'] == "xe":
                error = self.xe_loss(probabilities, targets)
            elif self.conf['loss'] == "oaxe":
                error, best_permutations = self.oaxe_loss(probabilities, targets)
                predictions = torch.gather(predictions, 1, best_permutations)
            else:
                raise ValueError(f"Unknown loss: {self.conf['loss']}")

            correct_batch, total_batch = calculate_accuracy(self.vocab, self.conf, predictions, targets)
            correct += correct_batch
            total += total_batch
            running_error += error.item() * total_batch

        valid_accuracy = correct / total
        valid_loss = running_error / total
        self.writer.add_scalar("valid accuracy",
                               valid_accuracy,
                               (self.current_epoch + 1) * len(self.valid_loader))
        self.writer.add_scalar("valid loss (average)",
                               valid_loss,
                               (self.current_epoch + 1) * len(self.valid_loader))
        print(f"EPOCH {self.current_epoch}")
        print(f"acc: {valid_accuracy}, loss: {valid_loss}")
        return valid_accuracy, valid_loss


    def oaxe_loss(self, probabilities_, targets):
        """
        Calculates Order-Agnostic Cross Entropy Loss
        """
        # probabilities has shape (max_sentence_length * batch_size, grammemes_in_vocab, max_label_length)
        # targets has shape (max_sentence_length * batch_size, max_label_length)
        probabilities = self.logsoftmax(probabilities_)

        best_permutations_ = torch.tensor(range(probabilities_.shape[2])).to(targets).repeat((probabilities_.shape[0], 1))
        for i in range(targets.shape[0]):  # for 1 sequence in a batch
            target_row = targets[i]
            n_nonpad = target_row.ne(0).sum()  # to exclude padding (0 is pad_id)
            if n_nonpad == 0:  # skip if row is padding itself
                continue
            if self.conf['same_length'] is False:
                n_nonpad -= 1 # in this case we should exclude eos token

            target_row = target_row[:n_nonpad] # remove padding
            probabilities_matrix = probabilities[i]  # probabilities for this sequence (grammemes_in_vocab, max_label_length)
            probabilities_matrix = probabilities_matrix[:, :n_nonpad] # remove probabilities on padded labels

            cost_matrix = -probabilities_matrix[target_row]  # get cost matrix for lsa
            cost_matrix_numpy = cost_matrix.detach().cpu().numpy()
            # permute because of how lsa works, we need rows indices, not column

            _, col_indices = lsa(cost_matrix_numpy)
            best_permutations_[i, :n_nonpad] = torch.tensor(col_indices).to(best_permutations_)

        best_permutations = best_permutations_[:, None, :]
        best_permutations = best_permutations.expand_as(probabilities)

        probabilities = torch.gather(probabilities, 2, best_permutations)

        smooth_loss = self.nll_loss(probabilities, targets)  # can be computed with regular negative log likelihood
        return smooth_loss, best_permutations_


def calculate_accuracy(vocabulary, conf, predictions, targets):
    """Metrics is a ratio of correctly predicted tags to total number of tags.

    All grammemes in a tag must be predicted correctly for it to count as correct.
    """

    n_total, n_correct = 0, 0
    masked_predictions = masked_select(predictions, vocabulary.vocab["grammeme-index"][conf['EOS']])

    iteration_mask = targets[:, 0].to(torch.bool)
    for tag, target in zip(masked_predictions[iteration_mask], targets[iteration_mask]):
        n_total += 1
        tag_nonzero = tag[tag.nonzero()]
        target_nonzero = target[target.nonzero()]
        equal = torch.equal(tag_nonzero, target_nonzero)
        n_correct += int(equal)

    return n_correct, n_total


def collate_batch(batch_w_sentences, label_length, pad_id=0, sos_id=1, eos_id=2):
    """Takes batch created with CustomDataset and performs padding.

    batch consists of indices of words, chars, and labels (grammemes). The returned batches are tensors.

    Args:
        batch_w_sentences (list): Single batch to be collated.
            Batch consists of tuples ((words, labels), raw_sentence) generated by CustomDataset.
        label_length (int): Length of labels. It is the same for all labels.
        pad_id (int, default 0): The id of the pad token in all dictionaries.
        sos_id (int, default 1): The id of the sos (start of sequence) token.
        eos_id (int, default 2): The id of the eos (end of sequence) token.

    Returns:
        tuple: (words_batch, chars_batch, labels_batch, raw_sentences).
            labels_batch is None if the dataset is for validation or test.
            All the elements in tuple have type torch.Tensor. Size of words_batch is (max_sentence_length, batch_size).
            Size of chars_batch is (batch_size * max_sentence_length, max_word_length). Size of labels_batch is
            (max_label_length, batch_size * max_sentence_length)
    """

    raw_sentences = [element[1] for element in batch_w_sentences] # raw sentences, used in valid_epoch
    batch = [element[0] for element in batch_w_sentences]

    sentences = [element[0] for element in batch]  # sentences is a list of all list of words
    max_sentence_length = max(map(lambda x: len(x), sentences))
    max_word_length = max([max(map(lambda x: len(x), sentence)) for sentence in sentences])

    words_batch = []
    chars_batch = []
    for words, _ in batch:
        words_indices = []
        chars_indices = []
        for word in words:
            word += [pad_id] * (max_word_length - len(word))  # id of the pad token must be 0
            words_indices += [word[0]]
            chars_indices += [word[1:]]
        words_indices += [pad_id] * (max_sentence_length - len(words))
        chars_indices += [[pad_id] * (max_word_length - 1)] * (max_sentence_length - len(words))
        # 1 subtracted from max_word_length because it includes id of the word itself

        words_batch += [words_indices]
        chars_batch += [chars_indices]

    words_batch = torch.tensor(words_batch, dtype=torch.long)
    words_batch = words_batch.transpose(1, 0)
    chars_batch = torch.tensor(chars_batch, dtype=torch.long)
    chars_batch = chars_batch.view(-1, chars_batch.shape[2])

    if batch[0][1][0]:
        labels_batch = collate_labels(batch, max_sentence_length, label_length, pad_id, sos_id, eos_id)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        labels_batch = labels_batch.view(-1, labels_batch.shape[2]).permute(1, 0)
    else:
        labels_batch = None
    return words_batch, chars_batch, labels_batch, raw_sentences


def collate_labels(batch, max_sentence_length, label_length, pad_id=0, sos_id=1, eos_id=2):
    if not label_length:
        tags = [element[1] for element in batch]  # tags is a list of all lists of grammemes
        max_label_length = 2 + max([max(map(lambda x: len(x), tag)) for tag in tags])  # +2 because of sos and eos tokens
    else:
        max_label_length = 1 + label_length # +1 because of sos token

    labels_batch = []
    for words, labels in batch:
        labels_indices = []
        for label in labels:
            label.insert(0, sos_id)  # id of the sos token must be 1
            if not label_length:
                label += [eos_id]  # id of the eos token must be 2
                label += [pad_id] * (max_label_length - len(label))
            labels_indices += [label]
        labels_indices += [[pad_id] * max_label_length] * (max_sentence_length - len(words))
        labels_batch += [labels_indices]
    return labels_batch


def subset_from_dataset(data, n):
    """Outputs first n entries from data (type Dataset) as another dataset."""
    subset = Subset(data, range(n)) if data else None
    return subset


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

    first_occurrence = (a == value).cumsum(dim=1)
    first_occurrence = first_occurrence.to(torch.bool)
    padding = torch.zeros((a.shape[0], 1), dtype=torch.bool).to(a.device)
    narrow = torch.narrow(first_occurrence, 1, 0, a.shape[1] - 1)
    mask = torch.hstack((padding, narrow))
    return a * (~mask)
