"""Contains method that predicts word tags for raw sentences."""

from model import Model
from utils import collate_batch

import torch


def predict(sentence, model_file="models/model_russian.pt"):
    """Uses saved model to assign tags to words for list of sentences and save them in a file.

    Args
        sentence (string): Sentence.
        model (string): File containing saved model parameters.
    """

    model = torch.load(model_file)
    vocab = model.data.vocab.vocab
    conf = model.conf

    sentence = sentence.split()
    words = []
    labels = []
    for word in sentence:
        word_ids = [vocab["word-index"].get(word, 1)]
        for char in word:
            word_ids += [vocab["char-index"].get(char, 1)]
        words += [word_ids]
        labels += [[]]

    batch = list(zip(words, labels))
    for w, c, l in collate_batch(batch):
        print(w, c, l)

    # print(model.data.vocab.vocab)


if __name__ == "__main__":
    predict("Чёрная машина едет по дороге .")
