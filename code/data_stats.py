from main import parse_arguments
from model.model import Model
from data_preparation.vocab import Vocab, get_vocab

import pickle

import fasttext


def show_statistics():
    conf = parse_arguments()
    conf['vocab_file'] = "./data_stats.pickle"
    train_vocab = get_vocab(conf, rewrite=True)
    print(len(train_vocab.vocab['word-index']))
    # conf['train_files'] = conf['test_files']
    # conf['vocab_file'] = "./data_stats_test.pickle"
    # print("\n========\n")
    # test_vocab = get_vocab(conf, rewrite=True)
    # print(len(test_wordforms), len(test_rawforms))
    # print(len(test_uniq_tags))

    # words_w_uniq_tags = 0
    # uniq_tags = set()
    # for word in test_word_types_tags.keys():
    #     for tag in test_word_types_tags[word]:
    #         if tag not in train_uniq_tags:
    #             words_w_uniq_tags += 1
    #             uniq_tags.add(tag)
    # assert uniq_tags - (test_uniq_tags - train_uniq_tags) == set()
    # print(f"uniq tags in test: {len(uniq_tags)}")
    # print(f"words with uniq tags in test: {words_w_uniq_tags}")

    # train_set = set(train_vocab.vocab["word-index"].keys())

    # oov_counter = 0
    # for word in test_rawforms:
    #     if word not in set(train_rawforms):
    #         oov_counter += 1
    #
    # print(f"oov words: {oov_counter}")

    # ft = fasttext.load_model("./data/wiki.ru.bin")
    # count = 0
    # for word in ft.words:
    #     if word in train_set:
    #         count += 1
    # print(f"% of words with pretrained embeddings: {count / (len(train_set) - 3)}")


if __name__ == "__main__":
    show_statistics()
