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
from pathlib import Path
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


def fill_conllu(conf, vocab, model, uniq_words, conll_input, conll_output):
    conll = pyconll.load_from_file(conll_input)

    print(f"{len(uniq_words)} oov words")
    print("Loading fastText...")
    ft = fasttext.load_model("./filled_conllu/cc.ru.300.bin")
    oov_pretrained_vocab = {}
    for word in ft.words:
        if word.lower() in uniq_words:
            oov_pretrained_vocab[word.lower()] = ft[word]

    print("Filling file...")
    progress_bar = tqdm(conll, total=len(conll), colour='#bbbbbb')
    for number, sentence in enumerate(progress_bar):
        word_list = []
        for token in sentence:
            if '.' not in token.id and '-' not in token.id:
                word_list += [token.form]

        words, labels = vocab.sentence_to_indices(word_list, None, training=False)
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
            other_grammemes = [g for g in label if "POS" not in g]
            pos_grammeme = next(iter([g for g in label if "POS" in g]), "POS=Did_not_predict")
            token.upos = pos_grammeme[4:]
            feats = defaultdict(list)
            for grammeme in other_grammemes:
                if grammeme == conf["EOS"]:
                    break
                key, feat = grammeme.split("=")
                feats[key] += [feat]
            token.feats = feats

    with open(conll_output, 'w+', encoding='utf-8') as f:
        conll.write(f)


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
    print(len(uniq_words))
    conll_true = pyconll.load_from_file(conll_true)
    conll_predicted = pyconll.load_from_file(conll_predicted)
    data = []
    for sentence_true, sentence_predicted in zip(conll_true, conll_predicted):
        neighbors_uniq = False
        for word_true in sentence_true:
            if '.' not in word_true.id and '-' not in word_true.id and word_true.form.lower() in uniq_words:
                neighbors_uniq = True
                break
        for word_true, word_predicted in zip(sentence_true, sentence_predicted):
            if '.' not in word_true.id and '-' not in word_true.id:
                label_true, label_predicted = [], []
                uniq = True if word_true.form.lower() in uniq_words else False

                label_true += ["POS=" + word_true.upos]
                for key_true in list(word_true.feats):
                    for feat_true in word_true.feats[key_true]:
                        label_true += [key_true + "=" + feat_true]
                tt = '|'.join(label_true)

                label_predicted += ["POS=" + word_predicted.upos]
                for key_predicted in list(word_predicted.feats):
                    for feat_predicted in word_predicted.feats[key_predicted]:
                        label_predicted += [key_predicted + "=" + feat_predicted]
                pt = '|'.join(label_predicted)

                data += [(word_true.form, tt, pt, neighbors_uniq, uniq, tt != pt)]

    df = pd.DataFrame(data=data, columns=['word', 'tt_full', 'pt_full', 'neighbors_uniq', 'uniq', 'err'])
    df['tt_pos'] = df.tt_full.str.extract(r'POS=([A-Z]+)', expand=False).fillna('')
    df['pt_pos'] = df.pt_full.str.extract(r'POS=([A-Z]+)', expand=False).fillna('')

    # set morph predictions
    df['tt_morph'] = df.tt_full.str.replace(r'(POS=[A-Za-z]+\|?)', '', regex=True)
    df['pt_morph'] = df.pt_full.str.replace(r'(POS=[A-Za-z]+\|?)', '', regex=True)

    return df


def analyze_df(df):
    correct_df = df[df.err == False]
    error_df = df[df.err == True]
    print(f"errors: {len(error_df)}")
    uniq_df = df[df.uniq == True]
    vocab_df = df[df.uniq == False]
    neighbors_uniq_df = df[df.neighbors_uniq == True]
    neighbors_vocab_df = df[df.neighbors_uniq == False]

    error_uniq_df = error_df[error_df.uniq == True]
    error_vocab_df = error_df[error_df.uniq == False]
    error_neighbors_uniq_df = error_df[error_df.neighbors_uniq == True]
    error_neighbors_vocab_df = error_df[error_df.neighbors_uniq == False]
    print(f"erroneous words that neighbor uniq: {len(error_neighbors_uniq_df)}, {len(error_neighbors_uniq_df) / len(neighbors_uniq_df)}")
    print(f"erroneous words that neighbor vocab: {len(error_neighbors_vocab_df)}, {len(error_neighbors_vocab_df) / len(neighbors_vocab_df)}")
    print(f"erroneous words that are uniq: {len(error_uniq_df)}, {len(error_uniq_df) / len(uniq_df)}")
    print(f"erroneous words that are vocab: {len(error_vocab_df)}, {len(error_vocab_df) / len(vocab_df)}")
    # category_df = {}
    # print(len(df))
    # for cat in ['Case', 'Gender', 'Number', 'VerbForm', 'Tense', 'Person', 'Degree', 'Aspect', 'Voice', 'Mood']:
    #     category_df[cat] = df.loc[df['tt_morph'].str.contains(cat)]
    #     category_df['error_' + cat] = category_df[cat][category_df[cat].err == True]
    #     print(f"errors for {cat} - {len(category_df['error_' + cat])}, "
    #           f"{len(category_df['error_' + cat]) / len(category_df[cat]):.3f}")
    # print(error_vocab_df)

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


def evaluate_models():
    trained_models_folder = "./trained_models"

    p = Path(trained_models_folder)

    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    for seed_number in range(5):
        model_paths = [str(x) for x in p.glob(f"*direct_order*/Russian-SynTagRus/seed_{seed_number}/model.pt")]
        vocab_paths = [str(x) for x in p.glob("*/Russian-SynTagRus/vocab.pickle")]

        for model_path, vocab_path in zip(model_paths, vocab_paths):
            model_name = model_path.split('\\')[1]
            print(f"{model_name} run {seed_number}")
            conf, vocab, model = load_model_vocab(model_path, vocab_path)
            # pathlib.PosixPath = temp
            uniq_words = get_uniq_words(conf, vocab, pyconll.load_from_file("./data/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu"))
            fill_conllu(conf, vocab, model, uniq_words, "./filled_conllu/Russian-SynTagRus/ru_syntagrus-ud-dev.conllu",
                        f"./filled_conllu/Russian-SynTagRus/ru_syntagrus-ud-dev_{model_name}_{seed_number}.conllu")
            df = construct_df("./filled_conllu/Russian-SynTagRus/ru_syntagrus-ud-dev.conllu",
                              f"./filled_conllu/Russian-SynTagRus/ru_syntagrus-ud-dev_{model_name}_{seed_number}.conllu", uniq_words)

            analyze_df(df)

            with open(f"./filled_conllu/results_dev_syntagrus_{seed_number}.txt", "w+") as file:
                file.write(f"\n{model_name}\n")
                results = calculate_accuracy_df(df)
                for key in results:
                    file.write(f"{key}: {results[key]:.5}\n")

    # model_path = "./trained_models/direct_order/Russian/seed_0/model.pt"
    # vocab_path = "./trained_models/direct_order/Russian/vocab.pickle"
    # conf, vocab, model = load_model_vocab(model_path, vocab_path)
    #
    # model.to(device)
    # model.eval()
    #
    # uniq_words = get_uniq_words(conf, vocab, pyconll.load_from_file("./data/UD_Russian/ru-ud-dev.conllu"))
    # fill_conllu(conf, vocab, model, uniq_words, "./filled_conllu/Russian/ru-ud-dev.conllu",
    #             f"./filled_conllu/Russian/ru-ud-dev_direct_order_0b.conllu")
    # df = construct_df("./filled_conllu/Russian/ru-ud-dev.conllu",
    #                   f"./filled_conllu/Russian/ru-ud-dev_direct_order_0b.conllu", uniq_words)
    # results = calculate_accuracy_df(df)
    # for key in results:
    #     print(f"{key}: {results[key]:.5}")


if __name__ == "__main__":
    evaluate_models()
# TODO: check that ABSOLUTELY everything is as in author's code and paper. Mark questionable places. Possible:
# pyconll loading data, metrics, padding learning, scale factor, loss function, default parameters in torch and tf
