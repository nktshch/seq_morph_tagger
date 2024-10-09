"""Contains method that predicts word tags for raw sentences."""

from trainer import collate_batch, calculate_accuracy
from model.model import Model
from data_preparation.vocab import Vocab

from collections import defaultdict
import numpy as np
import torch
from torch import cuda
import pickle
import pathlib
import fasttext
import pyconll
from tqdm import tqdm


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
    # TODO: remove order assignment for new models
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
    mask_embeddings = torch.zeros(words.shape, dtype=bool, device=device)
    if unk_id in words:
        print("Some words are not in vocab!")
        ft = fasttext.load_model("./Russian/cc.ru.300.bin")
        fasttext_embeddings = []
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
        correct, total = calculate_accuracy(vocab, conf, predictions, targets, mask_embeddings)
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


def fill_conllu(model_file, vocab_file, conll_input, conll_output):
    conf, state_dict = torch.load(model_file)
    # TODO: remove order assignment for new models
    conf['order'] = 'direct'
    with open(vocab_file, 'rb') as vf:
        vocab = pickle.load(vf)
        vocab.conf['order'] = 'direct'

    model = Model(conf, vocab)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    conll = pyconll.load_from_file(conll_input)

    print("Loading fastText...")
    ft = fasttext.load_model("./Russian/cc.ru.300.bin")
    progress_bar = tqdm(conll, total=len(conll), colour='#bbbbbb')
    for sentence in progress_bar:
        word_list = []
        for token in sentence:
            if '.' not in token.id and '-' not in token.id:
                word_list += [token.form]

        words, labels = vocab.sentence_to_indices(word_list, None)
        words, chars, _, __ = collate_batch([((words, labels), word_list)])
        words = words.to(device)
        chars = chars.to(device)

        unk_id = vocab.vocab["word-index"][conf["UNK"]]
        oov = None
        fasttext_embeddings = []
        mask_embeddings = torch.zeros(words.shape, dtype=bool, device=device)
        for i, word_index in enumerate(words):
            if word_index.item() == unk_id:
                if word_list[i] in ft.words:
                    fasttext_embeddings += [ft[word_list[i]]]
                    mask_embeddings[i] = True

        if torch.any(mask_embeddings):
            fasttext_embeddings = torch.tensor(np.array(fasttext_embeddings)).to(device)
            assert mask_embeddings.sum().item() == fasttext_embeddings.shape[0]
            oov = (fasttext_embeddings, mask_embeddings)

        with torch.no_grad():
            predictions, probabilities = model(words, chars, None, oov=oov)
        grammemes = predictions_to_grammemes(vocab.vocab, predictions.permute(1, 0))

        for label, token in zip(grammemes, sentence):
            token.upos = label[0][4:]
            feats = defaultdict(list)
            for grammeme in label[1:]:
                if grammeme == conf["EOS"]:
                    break
                key, feat = grammeme.split("=")
                feats[key] += [feat]
            token.feats = feats

    with open(conll_output, 'w+', encoding='utf-8') as f:
        conll.write(f)





if __name__ == "__main__":
    # sentence_ = "Безгачиха -- деревня в Бабушкинском районе Вологодской области .".split()
    # sentence_pyconll = pyconll.load.load_from_file("./Russian/test.conllu")[5]
    # predict("./Russian/seed_0/model.pt", "./Russian/vocab.pickle", sentence_, sentence_pyconll)

    fill_conllu("./Russian/seed_0/model.pt", "./Russian/vocab.pickle",
                "./Russian/test.conllu", "./Russian/test_filled.conllu")
