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
import re


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


def predict_sentence(sentence, output):
    conf, vocab, model = load_model_vocab("./trained_models/direct/Russian/seed_0/model.pt",
                                          "./trained_models/direct/Russian/vocab.pickle")
    word_list = re.findall(r"\b\w+(?:-\w+)*\b|[^\w\s]", sentence)
    print(word_list)

    train_words = set(vocab.vocab["word-index"])
    words = set()
    for word in word_list:
        if word.isdigit():
            words.add(conf['NUM'])
        else:
            words.add(word.lower())
    uniq_words = words - train_words

    print("Loading fastText...")
    ft = fasttext.load_model("./filled_conllu/wiki.ru.bin")
    oov_pretrained_vocab = {}
    for word in ft.words:
        if word.lower() in uniq_words:
            oov_pretrained_vocab[word.lower()] = ft[word]

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

    with open(output, "w+", encoding='utf-8') as txt:
        for label, word in zip(grammemes, word_list):
            s = []
            for g in label:
                if g == "$EOS$":
                    break
                s.append(g)
            txt.write(f"{word} - {s}\n")


def fill_conllu(conf, vocab, model, uniq_words, conll_input, conll_output):
    conll = pyconll.load_from_file(conll_input)
    txt_output = conll_output[:-6] + "txt"

    print(f"{len(uniq_words)} oov words")
    print("Loading fastText...")
    ft = fasttext.load_model("./filled_conllu/wiki.ru.bin")
    oov_pretrained_vocab = {}
    for word in ft.words:
        if word.lower() in uniq_words:
            oov_pretrained_vocab[word.lower()] = ft[word]

    print("Filling file...")
    progress_bar = tqdm(conll, total=len(conll), colour='#bbbbbb')
    with open(txt_output, "w+", encoding='utf-8') as txt:
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
                if '.' not in token.id and '-' not in token.id:
                    txt.write(f"{token.form} -")
                    for g in label:
                        if g == conf["EOS"]:
                            break
                        txt.write(f" {g}")
                    txt.write("\n")
                    # other_grammemes = [g for g in label if "POS" not in g]
                    # pos_grammeme = next(iter([g for g in label if "POS" in g]), "POS=Did_not_predict")
                    # token.upos = pos_grammeme[4:]
                    # feats = defaultdict(list)
                    # for grammeme in other_grammemes:
                    #     if grammeme == conf["EOS"]:
                    #         break
                    #     key, feat = grammeme.split("=")
                    #     feats[key] += [feat]
                    # token.feats = feats

    # with open(conll_output, 'w+', encoding='utf-8') as f:
    #     conll.write(f)


def analyze_order(txt_input, seed):
    with open(txt_input, "r", encoding='utf-8') as txt:
        lines = txt.readlines()
        # print(len(lines))

    data = []
    categories = set()
    for line in lines:
        features = {}
        word, grammemes = line.split(" - ")
        features["Form"] = word
        for i, g in enumerate(grammemes.split()):
            cat, val = g.split("=")
            features[cat + "_value"] = val
            features[cat + "_position"] = i
            categories.add(cat)
        data.append(features)

    categories = sorted(list(categories), key=len)
    columns = ["Form"] + [cat + "_value" for cat in categories] + [cat + "_position" for cat in categories]

    df = pd.DataFrame(data=data, columns=columns)
    print(f"TOTAL WORDS: {len(df)}")
    # df.to_excel(f"statistics_oa_{seed}.xlsx", na_rep="none")
    # print(df[df['POS_value'].notnull()])

    dataframes = [(df, "full")]

    df_pos_case = df[df[['POS_value', 'Case_value']].notnull().all(1)]
    dataframes.append((df_pos_case, "pos-case"))
    # print(df_pos_case[['POS_value', 'Case_value', 'POS_position', 'Case_position']])

    df_pos_animacy = df[df[['POS_value', 'Animacy_value']].notnull().all(1)]
    dataframes.append((df_pos_animacy, "pos-animacy"))
    # print(df_pos_animacy[['POS_value', 'Animacy_value', 'POS_position', 'Animacy_position']])

    df_tense_animacy = df[df[['Tense_value', 'Animacy_value']].notnull().all(1)]
    dataframes.append((df_tense_animacy, "tense-animacy"))
    # print(df_tense_animacy[['Form', 'POS_value', 'Tense_value', 'Animacy_value']])

    df_tense_case = df[df[['Tense_value', 'Case_value']].notnull().all(1)]
    dataframes.append((df_tense_case, "tense-case"))
    # print(df_tense_case[['Form', 'POS_value', 'Tense_value', 'Case_value']])

    # for dataframe, name in dataframes:
    #     for cat in categories:
    #         print(f"{cat}_mean {name}: {dataframe[cat + '_position'].mean()}")


def get_uniq_words(conf, vocab, conll):
    train_words = set(vocab.vocab["word-index"])
    test_words = set()
    for sentence in conll:
        for word in sentence:
            if '.' not in word.id and '-' not in word.id:
                if word.form.isdigit():
                    test_words.add(conf['NUM'])
                else:
                    test_words.add(word.form.lower())
    unseen_words = test_words - train_words
    return unseen_words


def construct_df(conll_true, conll_predicted, uniq_words):
    """Constructs DataFrame according to ground truth and model predictions.

    Args:
        conll_true (str): Ground truth file from UD library.
        conll_predicted (str): File filled by model.
        uniq_words (set): Set of oov words.

    """
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


def analyze_results():
    results_folder = "./filled_conllu"
    p = Path(results_folder)


    params = ['-'.join(x.stem.split('-')[1:-1]) for x in p.glob("results-*-0*")]
    keys = []
    for param in params:
        results = []
        seeds = [str(x) for x in p.glob(f"results-{param}-*")]
        for seed in seeds:
            with open(seed, "r") as s:
                lines = s.readlines()[1:]
                results.append([float(v.split(': ')[1]) for v in lines])
                if len(keys) == 0:
                    keys = [k.split(': ')[0] for k in lines]
        results = np.array(results)
        mean = results.mean(axis=0)
        std = results.std(axis=0)

        with open(f"{results_folder}/mean_std-{param}.txt", "w+") as file:
            file.write(f"{param}\n")
            for key, m_, s_ in zip(keys, mean, std):
                file.write(f"{key}: {m_:.5} ± {s_:.5}\n")


def evaluate_models(language="Russian", language_short="ru", dataset="train"):
    """Uses trained models to fill conllu files, then evaluates metrics and writes results to .txt files.

    Args:
        language (str, default 'Russian'): full language name as in UD library, but without 'UD_' prefix,
            e.g. 'Russian', 'Russian-SynTagRus'.
        language_short (str, default 'ru'): short language name as in conllu files, e.g. 'ru', 'ru_syntagrus'.
        dataset (str, default 'dev'): 'dev' or 'test'.
    """
    trained_models_folder = "./trained_models"

    p = Path(trained_models_folder)

    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    for seed_number in range(5):
        model_paths = [str(x) for x in p.glob(f"*order_agnostic/{language}/seed_{seed_number}/model.pt")]
        vocab_paths = [str(x) for x in p.glob(f"*/{language}/vocab.pickle")]

        for model_path, vocab_path in zip(model_paths, vocab_paths):
            model_name = model_path.split('\\')[1]
            print(f"{model_name} run {seed_number}")
            conf, vocab, model = load_model_vocab(model_path, vocab_path)
            # pathlib.PosixPath = temp

            conll_input = f"./filled_conllu/{language}/{language_short}-ud-{dataset}.conllu"
            conll_output = f"./filled_conllu/{language}/{language_short}-ud-{dataset}_{model_name}_{seed_number}.conllu"

            uniq_words = get_uniq_words(conf, vocab, pyconll.load_from_file(conll_input))
            fill_conllu(conf, vocab, model, uniq_words, conll_input, conll_output)
            df = construct_df(conll_input, conll_output, uniq_words)

            # analyze_df(df)

            # with open(f"./filled_conllu/results-{language_short}-{dataset}-{model_name}-{seed_number}.txt", "w+") as file:
            #     file.write(f"{language}, {dataset}, {model_name}\n")
            #     results = calculate_accuracy_df(df)
            #     for key in results:
            #         file.write(f"{key}: {results[key]:.5}\n")


if __name__ == "__main__":
    # evaluate_models(dataset="dev")
    # analyze_results()
    # predict_sentence("Каждый охотник желает знать, где сидит фазан.", "prediction.txt")
    for seed in range(1):
        print(f"====================\norder_agnostic run {seed}")
        analyze_order(f"filled_conllu/Russian/ru-ud-train_order_agnostic_{seed}.txt", seed=seed)
