"""Contains method that predicts word tags for raw sentences."""

from trainer import collate_batch, calculate_accuracy
from model.model import Model
from data_preparation.vocab import Vocab

import torch
from torch import cuda
import pickle


device = 'cuda' if cuda.is_available() else 'cpu'


def predict(model_file, vocab_file, sentence, sentence_pyconll=None):
    """Uses saved model to assign tags to words for list of sentences and save them in a file.

    Args:
        model_file (string): File containing saved model parameters, .pt.
        vocab_file (string): File containing vocab dictionary.
        sentence (list): List of words in a sentence.
        sentence_pyconll (list): If there is a sentence in pyconll format, it can be used to calculate accuracy.
    """

    conf, state_dict = torch.load(model_file)
    with open(vocab_file, 'rb') as vf:
        vocab = pickle.load(vf)

    model = Model(conf, vocab)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    words, labels = vocab.sentence_to_indices(sentence, sentence_pyconll)
    words, chars, labels = collate_batch([(words, labels)])

    words = words.to(device)
    chars = chars.to(device)
    if labels is not None:
        labels = labels.to(device)

    with torch.no_grad():
        predictions, probabilities = model(words, chars, labels)
    grammemes = predictions_to_grammemes(vocab.vocab, predictions.permute(1, 0))

    for word, tag in zip(sentence, grammemes):
        print(f"{word} - {tag}")

    # check accuracy if labels are available
    if labels is not None:
        targets = labels[1:]
        correct, total = calculate_accuracy(vocab, conf, predictions, targets)
        print(f"Correct: {correct}, accuracy: {correct / total}")


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
    sentence_ = "Безгачиха -- деревня в Бабушкинском районе Вологодской области .".split()
    import pyconll
    sentence_pyconll = pyconll.load.load_from_file("./model/test.conllu")[0]
    predict("./model/seed_0/model.pt", "./model/vocab.pickle", sentence_, sentence_pyconll)
