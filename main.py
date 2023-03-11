"""
The main file to run experiments.
"""
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModel

from data import get_dataloaders
from train import train
from utils import set_random_seeds
from hyperparameters import HyperParameters

import fire


def main(
    model_name_or_path='roberta-base',
    dataset_name='sst2',
    checkpoint_dir='saved_models',
    wandb_project='project',
    wandb_runname=None,
    optimizer='adam',
    seed=100,
    evaluate_test=True,
    **hyperparameters,
):
    hp = HyperParameters.from_dict(hyperparameters)
    set_random_seeds(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = os.path.dirname(checkpoint_dir)
    os.makedirs(base_path, exist_ok=True)

    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(base_path, 'train_log.log'), mode='a'),
        logging.StreamHandler(),
    ],
                        format='%(asctime)s:%(msecs)d|%(name)s|%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('Start Training!')

    # load model, tokenizer
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # load data
    dataset_kwargs = dict(tokenizer=tokenizer, dataset_name=dataset_name)
    dataloaders = get_dataloaders(batch_size=hp.batch_size,
                                  eval_batch_size=hp.eval_batch_size,
                                  **dataset_kwargs)

    # TODO: add more init & control here
    loss_fn = torch.nn.CrossEntropyLoss()  # TODO: change loss_fn.
    train(
        model,
        dataloaders,
        loss_fn,
        hp,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        wandb_project=wandb_project,
        wandb_runname=wandb_runname,
        evaluate_test=evaluate_test,
    )


if __name__ == '__main__':
    fire.Fire(main)
