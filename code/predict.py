"""Contains method that predicts word tags for raw sentences."""

from trainer import collate_batch, calculate_accuracy
from model.model import Model
from data_preparation.vocab import Vocab

import numpy as np
import torch
from torch import cuda
import pickle
import pathlib
import fasttext


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
    conf['order'] = 'direct'
    with open(vocab_file, 'rb') as vf:
        vocab = pickle.load(vf)
        vocab.conf['order'] = 'direct'

    model = Model(conf, vocab)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    words, labels = vocab.sentence_to_indices(sentence, sentence_pyconll)
    words, chars, labels, _ = collate_batch([((words, labels), sentence)])

    assert _[0] == sentence, f"{_[0]} is not equal to {sentence}"

    words = words.to(device)
    chars = chars.to(device)
    if labels is not None:
        labels = labels.to(device)

    unk_id = vocab.vocab["word-index"][conf["UNK"]]

    # print(words)
    oov = None
    if unk_id in words:
        print("Some words are not in vocab!")
        ft = fasttext.load_model("./Russian/cc.ru.300.bin")
        fasttext_embeddings = []
        mask_embeddings = torch.zeros(words.shape[0], dtype=bool, device=device)
        for i, word_index in enumerate(words):
            if word_index.item() == unk_id:
                print(f"'{sentence[i]}' is not in vocab")
                if sentence[i] in ft.words:
                    print(f"Found '{sentence[i]}' in fastText")
                    fasttext_embeddings += [ft[sentence[i]]]
                    mask_embeddings[i] = True

        fasttext_embeddings = torch.tensor(np.array(fasttext_embeddings)).to(device)
        assert mask_embeddings.sum().item() == fasttext_embeddings.shape[0]
        oov = (fasttext_embeddings, mask_embeddings)


    with torch.no_grad():
        predictions, probabilities = model(words, chars, None, oov=oov)
    grammemes = predictions_to_grammemes(vocab.vocab, predictions.permute(1, 0))

    for word, tag in zip(sentence, grammemes):
        print(word)
        print("PREDICTED")
        print(tag)
        print("\n")

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
    sentence_pyconll = pyconll.load.load_from_file("./Russian/test.conllu")[5]
    predict("./Russian/seed_0/model.pt", "./Russian/vocab.pickle", sentence_, sentence_pyconll)
