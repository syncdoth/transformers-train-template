"""
implementation of evaluation loop, loss, and metrics functions
"""
import torch
from tqdm import tqdm


def evaluate(model, eval_dataloader, loss_fn, device='cpu'):
    """ Evaluates a given model and dataset.
    Obtained from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/evaluate.py
    """
    model.eval()
    sample_count = 0
    running_loss = 0
    # TODO: implement other metrics

    with torch.no_grad():

        for i, batch in tqdm(enumerate(eval_dataloader),
                             position=0,
                             total=len(eval_dataloader),
                             desc='Evaluation'):
            inputs, label = (x.to(device) for x in batch)

            pred = model(inputs)
            loss = loss_fn(pred, label)

            sample_count += len(label)
            running_loss += loss.item() * len(label)  # smaller batches count less

        loss = running_loss / sample_count

    return loss
