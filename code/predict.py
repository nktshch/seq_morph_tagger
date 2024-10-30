"""Contains method that predicts word tags for raw sentences."""

from trainer import collate_batch, calculate_accuracy
from model.model import Model
from data_preparation.vocab import Vocab

from itertools import zip_longest
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import cuda
import pickle
import pathlib
import fasttext
import pyconll
from tqdm import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'


def load_model_vocab(model_file, vocab_file):
    conf, state_dict = torch.load(model_file)
    with open(vocab_file, 'rb') as vf:
        vocab = pickle.load(vf)

    model = Model(conf, vocab)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return conf, vocab, model

# TODO: rewrite or remove
# def predict(model_file, vocab_file, sentence, sentence_pyconll=None):
#     """Uses saved model to assign tags to words for list of sentences and save them in a file.
#
#     Args:
#         model_file (string): File containing saved model parameters, .pt.
#         vocab_file (string): File containing vocab dictionary.
#         sentence (list): List of words in a sentence.
#         sentence_pyconll (list): If there is a sentence in pyconll format, it can be used to calculate accuracy.
#     """
#
#     conf, state_dict = torch.load(model_file)
#     with open(vocab_file, 'rb') as vf:
#         vocab = pickle.load(vf)
#
#     model = Model(conf, vocab)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#
#     words, labels = vocab.sentence_to_indices(sentence, sentence_pyconll)
#     words, chars, labels, _ = collate_batch([((words, labels), sentence)])
#
#     assert _[0] == sentence, f"{_[0]} is not equal to {sentence}"
#
#     words = words.to(device)
#     chars = chars.to(device)
#     if labels is not None:
#         labels = labels.to(device)
#
#     unk_id = vocab.vocab["word-index"][conf["UNK"]]
#
#     # print(words)
#     oov = None
#     mask_embeddings = torch.zeros(words.shape, dtype=bool, device=device)
#     if unk_id in words:
#         print("Some words are not in vocab!")
#         ft = fasttext.load_model("./Russian/cc.ru.300.bin")
#         fasttext_embeddings = []
#         for i, word_index in enumerate(words):
#             if word_index.item() == unk_id:
#                 print(f"'{sentence[i].lower()}' is not in vocab")
#                 if sentence[i].lower() in ft.words:
#                     print(f"Found '{sentence[i].lower()}' in fastText")
#                     fasttext_embeddings += [ft[sentence[i].lower()]]
#                     mask_embeddings[i] = True
#
#         fasttext_embeddings = torch.tensor(np.array(fasttext_embeddings)).to(device)
#         assert mask_embeddings.sum().item() == fasttext_embeddings.shape[0]
#         oov = (fasttext_embeddings, mask_embeddings)
#
#
#     with torch.no_grad():
#         predictions, probabilities = model(words, chars, None, oov=oov)
#     grammemes = predictions_to_grammemes(vocab.vocab, predictions.permute(1, 0))
#
#     for word, tag in zip(sentence, grammemes):
#         print(word)
#         print("PREDICTED")
#         print(tag)
#         print("\n")
#
#     # check accuracy if labels are available
#     if labels is not None:
#         targets = labels[1:]
#         correct, total = calculate_accuracy(vocab, conf, predictions, targets, mask_embeddings)
#         print(f"Correct: {correct}, accuracy: {correct / total}")


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


def fill_conllu(conf, vocab, model, conll_input, conll_output):
    conll = pyconll.load_from_file(conll_input)

    uniq_words = get_uniq_words(conf, vocab, conll)
    print(f"{len(uniq_words)} oov words")
    print("Loading fastText...")
    ft = fasttext.load_model("./Russian/cc.ru.300.bin")
    oov_pretrained_vocab = {}
    for word in ft.words:
        if word.lower() in uniq_words:
            oov_pretrained_vocab[word.lower()] = ft[word]

    print("Filling file...")
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
                if word_list[i].lower() in oov_pretrained_vocab.keys():
                    fasttext_embeddings += [oov_pretrained_vocab[word_list[i].lower()]]
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


def conllu_accuracy(conll_true, conll_predicted):
    conll_true = pyconll.load_from_file(conll_true)
    conll_predicted = pyconll.load_from_file(conll_predicted)
    correct, total = 0, 0
    print("Calculating accuracy...")
    for sentence_true, sentence_predicted in zip(conll_true, conll_predicted):
        for word_true, word_predicted in zip(sentence_true, sentence_predicted):
            if '.' not in word_true.id and '-' not in word_true.id:
                total += 1
                if word_true.upos != word_predicted.upos:
                    break
                tag_true = []
                tag_predicted = []
                tag_true += [key + "=" + feat for key in list(word_true.feats) for feat in list(word_true.feats[key])]
                tag_predicted += [key + "=" + feat for key in list(word_predicted.feats) for feat in list(word_predicted.feats[key])]
                equal = set(tag_true) == set(tag_predicted)
                correct += int(equal)
    print(f"Accuracy: {correct / total}")
    return correct / total


def get_uniq_words(conf, vocab, conll):
    train_words = set(vocab.vocab["word-index"])
    test_words = set()
    test_with_digits = set()
    digits = set()
    for sentence in conll:
        for word in sentence:
            if '.' not in word.id and '-' not in word.id:
                if word.form.isdigit():
                    test_words.add(conf['NUM'])
                    digits.add(word.form)
                else:
                    test_words.add(word.form.lower())
                test_with_digits.add(word.form.lower())
    unseen_words = test_words - train_words
    return unseen_words


def construct_df(conll_true, conll_predicted, uniq_words):
    conll_true = pyconll.load_from_file(conll_true)
    conll_predicted = pyconll.load_from_file(conll_predicted)
    data = []
    for sentence_true, sentence_predicted in zip(conll_true, conll_predicted):
        for word_true, word_predicted in zip(sentence_true, sentence_predicted):
            if '.' not in word_true.id and '-' not in word_true.id:
                uniq = True if word_true.form.lower() in uniq_words else False
                data += [(word_true.form, "POS=" + word_true.upos, "POS=" + word_predicted.upos, uniq, word_true.upos != word_predicted.upos)]
                for key_true in list(word_true.feats):
                    # zip_longest should be used instead
                    for feat_true, feat_predicted in zip(word_true.feats[key_true], word_predicted.feats.get(key_true, [])):
                        if feat_predicted is None:
                            data += [(word_true.form, key_true + "=" + feat_true, '', uniq, True)]
                        else:
                            data += [(word_true.form, key_true + "=" + feat_true, key_true + "=" + feat_predicted, uniq, feat_true != feat_predicted)]


    df = pd.DataFrame(data=data, columns=['word', 'tt_full', 'pt_full', 'uniq', 'err'])
    df['tt_pos'] = df.tt_full.str.extract(r'POS=([A-Z]+)', expand=False).fillna('')
    df['pt_pos'] = df.pt_full.str.extract(r'POS=([A-Z]+)', expand=False).fillna('')

    # set morph predictions
    df['tt_morph'] = df.tt_full.str.replace(r'(\|POS=[A-Za-z]+\|)', '|').str.replace(r'(\|?POS=[A-Za-z]+\|?)', '')
    df['pt_morph'] = df.pt_full.str.replace(r'(\|POS=[A-Za-z]+\|)', '|').str.replace(r'(\|?POS=[A-Za-z]+\|?)', '')

    return df


def calculate_accuracy_df(df):
    from sklearn.metrics import accuracy_score
    return OrderedDict(
        # full tag  accuracy
        acc_full_all=accuracy_score(df.tt_full, df.pt_full),
        acc_full_oov=accuracy_score(df[df.uniq == True].tt_full, df[df.uniq == True].pt_full),
        acc_full_voc=accuracy_score(df[df.uniq == False].tt_full, df[df.uniq == False].pt_full),

        # pos accuracy
        acc_pos_all=accuracy_score(df.tt_pos, df.pt_pos),
        acc_pos_oov=accuracy_score(df[df.uniq == True].tt_pos, df[df.uniq == True].pt_pos),
        acc_pos_voc=accuracy_score(df[df.uniq == False].tt_pos, df[df.uniq == False].pt_pos),

        # morphology tag accuracy
        acc_morph_all=accuracy_score(df.tt_morph, df.pt_morph),
        acc_morph_oov=accuracy_score(df[df.uniq == True].tt_morph, df[df.uniq == True].pt_morph),
        acc_morph_voc=accuracy_score(df[df.uniq == False].tt_morph, df[df.uniq == False].pt_morph)
    )


if __name__ == "__main__":
    # sentence_ = "Безгачиха -- деревня в Бабушкинском районе Вологодской области .".split()
    # sentence_pyconll = pyconll.load.load_from_file("./Russian/test.conllu")[5]
    # predict("./Russian/seed_0/model.pt", "./Russian/vocab.pickle", sentence_, sentence_pyconll)



    # conllu_accuracy("./Russian/train.conllu", "./Russian/train_filled.conllu")

    for i in range(5):
        conf_, vocab_, model_ = load_model_vocab(f"./Russian/seed_{i}/model.pt", "./Russian/vocab.pickle")
        uniq_words_ = get_uniq_words(conf_, vocab_, pyconll.load_from_file("./Russian/test.conllu"))
        fill_conllu(conf_, vocab_, model_,
                    "./Russian/test.conllu", f"./Russian/test_{i}_filled.conllu")
        df_ = construct_df("./Russian/test.conllu", f"./Russian/test_{i}_filled.conllu", uniq_words_)
        print(calculate_accuracy_df(df_))


