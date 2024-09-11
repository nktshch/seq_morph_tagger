"""Pipeline of the model."""

from config import configurate
from vocab import Vocab
from dataset import CustomDataset
from model import Model
from trainer import Trainer

import json
import torch
import numpy as np


def main():
    with open('code/config.json', 'r') as json_file:
        conf = json.load(json_file)
    configurate(conf)

    # for key in conf:
    #     print(f"{key} : {conf[key]}")

    print(f"Training model for {conf['number_of_runs']}")
    for run_number in range(conf['number_of_runs']):
        torch.backends.cudnn.deterministic = True
        torch.random.manual_seed(run_number)
        torch.cuda.manual_seed(run_number)
        np.random.seed(run_number)

        vocabulary = Vocab(conf)
        train_data = CustomDataset(conf, vocabulary, conf['train_directory'],
                                   sentences_pickle=conf['train_sentences_pickle'], training_set=True)
        valid_data = CustomDataset(conf, vocabulary, conf['valid_directory'],
                                   sentences_pickle=conf['valid_sentences_pickle'], training_set=False)
        test_data = CustomDataset(conf, vocabulary, conf['test_directory'],
                                  sentences_pickle=conf['test_sentences_pickle'], training_set=False)
        model = Model(conf, train_data).to(conf['device'])
        trainer = Trainer(conf, model, valid_data, test_data, run_number=run_number, subset_size=0).to(conf['device'])
        trainer.epoch_loops()

        print("Training complete")


if __name__ == "__main__":
    main()
