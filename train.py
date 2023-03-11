"""
implementation of the training loop. column_width=120
"""
import gc
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import wandb

from evaluate import evaluate
from hyperparameters import HyperParameters


def train(
    model: nn.Module,
    dataloader,
    loss_fn,
    hp: HyperParameters,
    optimizer='adam',
    device='cpu',
    checkpoint_dir='saved_models',
    wandb_project='project',
    wandb_runname=None,
    evaluate_test=True,
):
    """ Trains a given model and dataset.

    obtained and adapted from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/train.py
    """
    # optimizers
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=hp.lr,
                              weight_decay=hp.weight_decay,
                              momentum=0.9)
    else:
        raise NotImplementedError(f'{hp.optimizer} not setup.')

    num_steps_per_epoch = len(dataloader['train'])

    # lr schedulers
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=hp.num_warmup_steps,
                                                   num_training_steps=num_steps_per_epoch *
                                                   hp.epochs)

    epoch_valid_loss = None
    best_valid_loss = np.inf

    if not wandb_runname:
        wandb_runname = str(round(time.time() * 1000))
    experiment = wandb.init(project=wandb_project, name=wandb_runname, config=hp)

    since = time.time()
    for epoch in range(1, hp.epochs + 1):
        model.train()
        sample_count = 0
        running_loss = 0

        logging.info(f'\nEpoch {epoch}/{hp.epochs}:\n')
        for i, batch in tqdm(enumerate(dataloader['train']),
                             position=0,
                             total=num_steps_per_epoch,
                             desc=f'Train Epoch {epoch}'):
            inputs, label = (x.to(device) for x in batch)

            pred = model(inputs)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            sample_count += len(label)
            running_loss += loss.item() * len(label)  # smaller batches count less

            # wandb log
            log_items = {
                "loss": loss.item(),
                "LR": lr_scheduler.get_last_lr()[0],
            }
            experiment.log(log_items)

        epoch_train_loss = running_loss / sample_count
        log_items = {"Train Epoch Loss": epoch_train_loss.item()}
        experiment.log(log_items)

        epoch_valid_loss = evaluate(model, dataloader['valid'], loss_fn, device=device)
        log_items = {"valid Epoch Loss": epoch_valid_loss.item()}
        experiment.log(log_items)

        logging.info(
            f'\n[Train] loss: {epoch_train_loss:.4f} | [Valid] loss: {epoch_valid_loss:.4f}')

        # save model and early stopping
        if epoch_valid_loss < best_valid_loss:
            best_epoch = epoch + 1
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
        epoch_valid_loss = None  # reset loss

        gc.collect()  # release unreferenced memory

    time_elapsed = time.time() - since
    logging.info(f'\nTraining time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if evaluate_test:
        logging.info('evaluating on test set now')
        model.load_state_dict(torch.load(checkpoint_dir))  # load best model

        test_loss = evaluate(model, dataloader['test'], loss_fn, device=device)

        log_items = {"Test Loss": test_loss.item()}
        experiment.log(log_items)

        logging.info(f'\nBest [Valid] | epoch: {best_epoch} - loss: {best_valid_loss:.4f} ')
        logging.info(f'[Test] loss {test_loss:.4f}')
