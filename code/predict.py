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
    """Predicts morphological categories for sentence and outputs to txt file"""

    # add arguments for model
    conf, vocab, model = load_model_vocab("./trained_models/order_agnostic/Russian-SynTagRus/seed_0/model.pt",
                                          "./trained_models/order_agnostic/Russian-SynTagRus/vocab.pickle")
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
            tag = []
            for g in label:
                if g == "$EOS$":
                    break
                tag.append(g)
            txt.write(f"{word} - {tag}\n")


def fill(conf, vocab, model, uniq_words, conll_input, conll_output):
    """Given file in conllu format, creates another with morphological categories filled.

    Also, creates txt file with the same name where words and their tags are written line by line.
    It is then used to retrieve positions of predicted grammemes.
    """

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

            index = 0
            for token in sentence:
                if '.' not in token.id and '-' not in token.id:
                    label = grammemes[index]
                    index += 1
                    txt.write(f"{token.form} -")
                    for g in label:
                        if g == conf["EOS"]:
                            break
                        txt.write(f" {g}")
                    txt.write("\n")
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


def analyze_order(language, language_short, dataset, seed=0):
    """
    Method creates files and DataFrames that contain information about order of predictions.
    See inline comments for details.
    For arguments meanings, see evaluate_models().
    """

    # 1. load txt file created in fill_conllu()
    txt_input = f"filled_conllu/{language}/{language_short}-ud-{dataset}-order_agnostic-{seed}.txt"
    with open(txt_input, "r", encoding='utf-8') as txt:
        lines = txt.readlines()
        # print(len(lines))

    # 2. load DataFrame created in evaluate_models() to include accuracy for orders
    df_loaded = pd.read_csv(f"{language}-{dataset}-{seed}.csv")
    df_word_err = df_loaded[["word", "err"]]

    # 3. create DataFrame where each row corresponds to one token and specifies info about grammemes and their positions
    # columns "Full_tag", "Full_cat", "Set_g" and "Set_c" contain info that can be retrieved from other columns,
    # and are needed to avoid repetitive computations
    data_cats = []
    categories = set()
    for index, line in enumerate(lines):
        features = {}
        word, grammemes = line.split(" - ")
        assert word == df_word_err["word"].iloc[index], f'{index}, {word}, {df_word_err["word"].iloc[index]}'
        features["Form"] = word
        features["Error"] = df_word_err["err"].iloc[index]
        features["Full_tag"] = grammemes[:-1] # grammemes in order of predictions
        g_split = grammemes.split()
        c_split = [i.split('=')[0] for i in g_split]
        features["Full_cat"] = ' '.join(c_split) # morphological categories in order of prediction
        features["Set_g"] = frozenset(g_split) # set of grammemes
        features["Set_c"] = frozenset(c_split) # set of morphological categories
        for i, g in enumerate(g_split):
            cat, val = g.split("=")
            # other columns can be reconstructed from following:
            features[cat + "_value"] = val # columns for categories values
            features[cat + "_position"] = i # columns for categories positions
            categories.add(cat)
        data_cats.append(features)

    categories = sorted(list(categories), key=len)
    values = [cat + "_value" for cat in categories]
    positions = [cat + "_position" for cat in categories]
    columns = ["Form", "Error", "Full_tag", "Full_cat", "Set_g", "Set_c"] + values + positions

    df = pd.DataFrame(data=data_cats, columns=columns)
    print(f"TOTAL WORDS: {len(df)}")

    # the loops below contain repetitive code, consider refactoring to a separate method with arguments

    # 4.1 write to file possible orders for sets of grammemes, i.e. categories AND values
    counter = 0
    with open(f"filled_conllu/_analyze_order/unique_tags-{language_short}-{dataset}-{seed}.txt", "w+", encoding='utf-8') as txt:
        dfs_by_tags = df.groupby("Set_g", dropna=False)
        print(f"TOTAL TAG GROUPS: {dfs_by_tags.ngroups}")
        txt.write(f"TOTAL TAG GROUPS: {dfs_by_tags.ngroups}\n\n")
        for tag, dataframe in dfs_by_tags:
            current_set = ' '.join(str(g) for g in sorted(tag, key=len))
            txt.write(f"Current set: {current_set} ({len(dataframe)})\n")
            counter += len(dataframe)
            subgroup_sizes = dataframe.groupby(["Full_tag", "Error"]).size()
            group_sizes = dataframe.groupby("Full_tag").size()
            relative_sizes = subgroup_sizes / group_sizes
            for v, c in group_sizes.items():
                try:
                    r = relative_sizes[(v, False)]
                    txt.write(f"\t{v} ({c}, {r:.4})\n")
                except KeyError:
                    txt.write(f"\t{v} ({c}, {0.0})\n")
    assert counter == len(df), f"{counter}, {len(df)}"

    # 4.2 write to file possible orders for sets of categories, i.e. only categories
    data_cats = []
    counter = 0
    with open(f"filled_conllu/_analyze_order/unique_cats-{language_short}-{dataset}-{seed}.txt", "w+", encoding='utf-8') as txt:
        dfs_by_cats = df.groupby("Set_c", dropna=False)
        print(f"TOTAL CAT GROUPS: {dfs_by_cats.ngroups}")
        txt.write(f"TOTAL CAT GROUPS: {dfs_by_cats.ngroups}\n\n")
        for cat, dataframe in dfs_by_cats:
            current_set = ' '.join(str(g) for g in sorted(cat, key=len))
            txt.write(f"Current set: {current_set} ({len(dataframe)})\n")
            counter += len(dataframe)
            subgroup_sizes = dataframe.groupby(["Full_cat", "Error"]).size()
            group_sizes = dataframe.groupby("Full_cat").size()
            relative_sizes = subgroup_sizes / group_sizes
            for v, c in group_sizes.items():
                try:
                    r = relative_sizes[(v, False)]
                    txt.write(f"\t{v} ({c}, {r:.4})\n")
                    data_cats += [(current_set, v, c, r)]
                except KeyError:
                    txt.write(f"\t{v} ({c}, {0.0})\n")
                    data_cats += [(current_set, v, c, 0.0)]
    assert counter == len(df), f"{counter}, {len(df)}"
    df_order = pd.DataFrame(data=data_cats, columns=["Set", "Order", "Count", "Accuracy"])
    df_order.to_csv(f"filled_conllu/_analyze_order/unique_cats-{language_short}-{dataset}-{seed}.csv", encoding="utf-8")

    # 4.3 for every POS value, write to file orders for sets that contain that POS
    data_pos = []
    counter = 0
    with open(f"filled_conllu/_analyze_order/unique_pos-{language_short}-{dataset}-{seed}.txt", "w+", encoding='utf-8') as txt:
        dfs_by_pos = df.groupby("POS_value", dropna=False)
        print(f"TOTAL POS VALUES: {dfs_by_pos.ngroups}")
        txt.write(f"TOTAL POS VALUES: {dfs_by_pos.ngroups}\n\n")
        for pos_value, dataframe in dfs_by_pos:
            txt.write(f"POS: {pos_value} ({len(dataframe)})\n")
            counter += len(dataframe)
            subgroup_sizes = dataframe.groupby(["Full_cat", "Error"]).size()
            group_sizes = dataframe.groupby("Full_cat").size().sort_values(ascending=False)
            relative_sizes = subgroup_sizes / group_sizes
            for v, c in group_sizes.items():
                try:
                    r = relative_sizes[(v, False)]
                    txt.write(f"\t{v} ({c}, {r:.4})\n")
                    data_pos += [(pos_value, v, c, r)]
                except KeyError:
                    txt.write(f"\t{v} ({c}, {0.0})\n")
                    data_pos += [(pos_value, v, c, 0.0)]
    assert counter == len(df), f"{counter}, {len(df)}"
    df_pos = pd.DataFrame(data=data_pos, columns=["POS", "Order", "Count", "Accuracy"])
    df_pos.to_csv(f"filled_conllu/_analyze_order/unique_pos-{language_short}-{dataset}-{seed}.csv",
                  encoding="utf-8", na_rep="nan")

    # 5. write to file average positions of categories
    with open(f"filled_conllu/_analyze_order/positions-{language_short}-{dataset}-{seed}.txt", "w+", encoding='utf-8') as txt:
        txt.write(f"Total words: {len(df)}\n\n")
        txt.write("Category (avg position/occurences)\n")
        txt.write("----------------------------------\n")
        stats = []
        for cat in categories:
            stats.append([cat, (df[cat + '_position'].mean().round(5), df[cat + '_position'].count())])
        stats.sort(key=lambda x: x[1][0])
        for i, j in stats:
            txt.write(f"{i} {j}\n")


def visualize_order(language, language_short, dataset, seed=0):
    """Method uses Plotly to visualize orders of predictions."""

    import plotly.express as px
    csv_input = f"filled_conllu/_analyze_order/unique_pos-{language_short}-{dataset}-{seed}.csv"
    df = pd.read_csv(csv_input, na_values="nan")
    # just to make sure that DataFrame was saved and loaded correctly (compare to printouts from analyze_order())
    print(f'Assert {df["POS"].nunique(dropna=False)} unique groups')
    print(f'Assert {df["Count"].sum()} total words')
    print(f'Assert {((df["Count"] * df["Accuracy"]).sum() / df["Count"].sum()):.5} accuracy') # refer to txt files

    fig = px.scatter(df, x="Count", y="Accuracy", color="POS", hover_data="Order", log_x=True, log_y=True)
    fig.show()


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


def calculate_mean_std(language_short="ru"):
    results_folder = "./filled_conllu"
    p = Path(results_folder)

    params = ['-'.join(x.stem.split('-')[1:-1]) for x in p.glob(f"results-{language_short}-*-4*")]
    keys = []
    for param in params:
        results = []
        seeds = [str(x) for x in p.glob(f"results-{param}-*")]
        for seed in seeds:
            with open(seed, "r") as f:
                lines = f.readlines()[1:]
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


def evaluate_models(language="Russian", language_short="ru", dataset="dev", save_df=True):
    """Uses trained models to fill conllu files, then evaluates metrics and writes results to .txt files.

    Args:
        language (str, default 'Russian'): full language name as in UD library, but without 'UD_' prefix,
            e.g. 'Russian', 'Russian-SynTagRus'.
        language_short (str, default 'ru'): short language name as in conllu files, e.g. 'ru', 'ru_syntagrus'.
        dataset (str, default 'dev'): 'dev' or 'test'.
        save_df (bool, default True): determines whether to save DataFrame of all words to csv.
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
            print(f"{language}, {dataset}, {model_name}, run {seed_number}")
            conf, vocab, model = load_model_vocab(model_path, vocab_path)
            # pathlib.PosixPath = temp

            conll_input = f"./filled_conllu/{language}/{language_short}-ud-{dataset}.conllu"
            conll_output = f"./filled_conllu/{language}/{language_short}-ud-{dataset}-{model_name}-{seed_number}.conllu"

            uniq_words = get_uniq_words(conf, vocab, pyconll.load_from_file(conll_input))
            print("Filling files...")
            fill(conf, vocab, model, uniq_words, conll_input, conll_output)
            print("Files filled")

            print("Constructing DataFrame...")
            df = construct_df(conll_input, conll_output, uniq_words)
            if save_df:
                df.to_csv(f"{language}-{dataset}-{seed_number}.csv", na_rep="NULL", encoding="utf-8")
            print("DataFrame done")

            with open(f"./filled_conllu/results-{language_short}-{dataset}-{model_name}-{seed_number}.txt", "w+") as file:
                file.write(f"{language}, {dataset}, {model_name}\n")
                results = calculate_accuracy_df(df)
                for key in results:
                    file.write(f"{key}: {results[key]:.5}\n")


if __name__ == "__main__":
    lang, lang_short, dset = ("Russian", "ru", "dev")
    # lang, lang_short, dset = ("Russian-SynTagRus", "ru_syntagrus", "dev")

    # evaluate_models(language=lang, language_short=lang_short, dataset=dset, save_df=True)
    # calculate_mean_std(language_short=lang_short)
    # predict_sentence('Каждый охотник желает знать, где сидит фазан.', "prediction.txt")
    for s in range(5):
        print(f"====================\n"
              f"Analyze order: run {s}")
        analyze_order(language=lang, language_short=lang_short, dataset=dset, seed=s)
    for s in range(5):
        print(f"======================\n"
              f"Visualize order: run {s}")
        visualize_order(language=lang, language_short=lang_short, dataset=dset, seed=s)

    # TODO: calculate number of orders (average among all sets)
    # split groups further: by gender, animacy or other
    # add train set and look only at correct predictions
