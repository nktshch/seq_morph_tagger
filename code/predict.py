"""Contains method that predicts word tags for raw sentences."""

from model import Model

import torch
from torch import cuda


# device = 'cuda' if cuda.is_available() else 'cpu'


def predict(sentence, model_file="models/model_russian.pt"):
    """Uses saved model to assign tags to words for list of sentences and save them in a file.

    Args:
        sentence (list): List of words in a sentence.
        model_file (string): File containing saved model parameters.
    """

    model = torch.load(model_file)
    vocab = model.data.vocab.vocab
    device = model.conf['device']

    words, chars, labels = collate_single(sentence, vocab)

    words = words.to(device)
    chars = chars.to(device)
    labels = labels.to(device)

    predictions, probabilities = model(words, chars, labels)

    grammemes = predictions_to_grammemes(vocab, predictions.permute(1, 0))

    for word, tag in zip(sentence, grammemes):
        print(f"{word} - {tag}")


def collate_single(sentence, vocab, pad_id=0, sos_id=1):
    """Collates single sentence with no labels. This is essentially CustomDataset with 1 sentence followed by
    collate_batch with batch size 1.

    Used in predict method that takes raw sentence and predicts grammemes for all words.

    Args:
        sentence (list): Sentence as a list of words.
        vocab (vocab): Vocabulary from Vocab class. Used to map chars and words to indices.
        pad_id (int, default 0): The id of the pad token in all dictionaries.
        sos_id (int, default 1): The id of the sos (start of sequence) token.

    Returns:
        tuple: (words, chars, labels). All of them have type torch.Tensor. Size of words is (sentence_length, 1).
            Size of chars is (sentence_length, max_word_length). Size of labels is (1, sentence_length)

    """

    words = []
    chars = []
    labels = []
    for word in sentence:
        words += [vocab["word-index"].get(word, 1)]
        char_ids = []
        for char in word:
            char_ids += [vocab["char-index"].get(char, 1)]
        chars += [char_ids]
        labels += [[]]

    max_word_length = max(map(lambda x: len(x), chars))
    sentence_length = len(words)

    for chars_indices in chars:
        chars_indices += [pad_id] * (max_word_length - len(chars_indices))


    words = torch.tensor(words, dtype=torch.long)
    words = words[:, None]
    chars = torch.tensor(chars, dtype=torch.long)
    labels = sos_id * torch.ones(1, sentence_length, dtype=torch.long)

    return words, chars, labels


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
