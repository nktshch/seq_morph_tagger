"""Contains method that predicts word tags for raw sentences."""

from trainer import collate_batch
from model.model import Model
from data_preparation.vocab import Vocab

import torch


# device = 'cuda' if cuda.is_available() else 'cpu'


def predict(sentence, sentence_pyconll=None, model_file="models/small_model.pt"):
    """Uses saved model to assign tags to words for list of sentences and save them in a file.

    Args:
        sentence (list): List of words in a sentence.
        sentence_pyconll (list): If there is a sentence in pyconll format, it can be used to calculate accuracy.
        model_file (string): File containing saved model parameters.
    """

    conf, vocab, state_dict = torch.load(model_file)
    model = Model(conf, vocab)
    model.load_state_dict(state_dict)
    device = conf['device']
    model.to(device)
    model.eval()

    words, labels = vocab.sentence_to_indices(sentence, sentence_pyconll)
    words, chars, labels = collate_batch([(words, labels)])

    words = words.to(device)
    chars = chars.to(device)
    if labels:
        labels = labels.to(device)

    predictions, probabilities = model(words, chars, None)
    grammemes = predictions_to_grammemes(vocab.vocab, predictions.permute(1, 0))

    for word, tag in zip(sentence, grammemes):
        print(f"{word} - {tag}")

    # check accuracy if labels are available
    if labels:
        targets = labels[1:]
        correct, total = calculate_accuracy(vocab, conf, predictions, targets)
        print(f"Correct: {correct}, accuracy: {correct / total}")


# def test_model(self):
#     self.model.eval()
#
#     # code similar to train_epoch
#     progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), colour='#ffbbbb')
#     running_error = 0.0
#     correct, total = 0, 0
#     for iteration, (words_batch, chars_batch, labels_batch) in progress_bar:
#         words_batch = words_batch.to(self.conf['device'])
#         chars_batch = chars_batch.to(self.conf['device'])
#         labels_batch = labels_batch.to(self.conf['device'])
#
#         predictions, probabilities = self.model(words_batch, chars_batch, None)
#         targets = labels_batch[1:]  # slice is taken to ignore SOS token
#         probabilities = probabilities[:len(targets)]
#
#         error = self.loss(probabilities.permute(1, 2, 0), targets.permute(1, 0))
#         running_error += error.item()
#
#         correct_batch, total_batch = calculate_accuracy(self.vocab, self.conf, predictions, targets)
#         correct += correct_batch
#         total += total_batch
#     valid_accuracy = correct / total
#     valid_loss = running_error / (self.current_epoch + 1) * len(self.valid_loader)
#     self.writer.add_scalar("valid accuracy",
#                            valid_accuracy,
#                            (self.current_epoch + 1) * len(self.valid_loader))
#     self.writer.add_scalar("valid loss",
#                            valid_loss,
#                            (self.current_epoch + 1) * len(self.valid_loader))
#     return valid_accuracy, valid_loss


# def collate_single(sentence, vocab, pad_id=0, sos_id=1):
#     """Collates single sentence with no labels. This is essentially CustomDataset with 1 sentence followed by
#     collate_batch with batch size 1.
#
#     Used in predict method that takes raw sentence and predicts grammemes for all words.
#
#     Args:
#         sentence (list): Sentence as a list of words.
#         vocab (vocab): Vocabulary from Vocab class. Used to map chars and words to indices.
#         pad_id (int, default 0): The id of the pad token in all dictionaries.
#         sos_id (int, default 1): The id of the sos (start of sequence) token.
#
#     Returns:
#         tuple: (words, chars, labels). All of them have type torch.Tensor. Size of words is (sentence_length, 1).
#             Size of chars is (sentence_length, max_word_length). Size of labels is (1, sentence_length)
#
#     """
#
#     words = []
#     chars = []
#     labels = []
#     for word in sentence:
#         words += [vocab["word-index"].get(word, 1)]
#         char_ids = []
#         for char in word:
#             char_ids += [vocab["char-index"].get(char, 1)]
#         chars += [char_ids]
#         labels += [[]]
#
#     max_word_length = max(map(lambda x: len(x), chars))
#     sentence_length = len(words)
#
#     for chars_indices in chars:
#         chars_indices += [pad_id] * (max_word_length - len(chars_indices))
#
#
#     words = torch.tensor(words, dtype=torch.long)
#     words = words[:, None]
#     chars = torch.tensor(chars, dtype=torch.long)
#     labels = sos_id * torch.ones(1, sentence_length, dtype=torch.long)
#
#     return words, chars, labels


def predictions_to_grammemes(vocabulary, predictions):
    """Turns indices of predictions produced by decoder into actual grammemes (strings).

    Args:
        vocabulary (dict): Vocabulary from Vocab class.
        predictions (torch.Tensor): 2D Tensor containing indices.

    Returns:
        list: List of lists of predicions.
    """

    tags = []
    for tag_indices in predictions:
        tag = []
        for grammeme_index in tag_indices:
            tag += [vocabulary['index-grammeme'][grammeme_index.item()]]
        tags += [tag]
    return tags


if __name__ == "__main__":
    sentence_ = "Синяя машина будет ехать по дороге .".split()
    predict(sentence_)
