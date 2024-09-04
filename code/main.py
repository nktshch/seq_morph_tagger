"""Pipeline of the model."""

import json

from config import configurate
from vocab import Vocab
from dataset import CustomDataset
from model import Model
from trainer import Trainer

def main():
    with open('code/config.json', 'r') as json_file:
        conf = json.load(json_file)
    configurate(conf)

    # for key in conf:
    #     print(f"{key} : {conf[key]}")

    vocabulary = Vocab(conf)
    train_data = CustomDataset(conf, vocabulary, conf['train_directory'],
                               sentences_pickle=conf['train_sentences_pickle'], training_set=True)
    valid_data = CustomDataset(conf, vocabulary, conf['valid_directory'],
                               sentences_pickle=conf['valid_sentences_pickle'], training_set=False)
    test_data = CustomDataset(conf, vocabulary, conf['test_directory'],
                              sentences_pickle=conf['test_sentences_pickle'], training_set=False)
    model = Model(conf, train_data).to(conf['device'])
    trainer = Trainer(conf, model, valid_data, test_data, subset_size=0).to(conf['device'])
    trainer.epoch_loops()


if __name__ == "__main__":
    main()
