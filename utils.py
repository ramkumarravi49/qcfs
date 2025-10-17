import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
from Models import IF

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

# def train(model, device, train_loader, criterion, optimizer, T):
#     running_loss = 0
#     model.train()
#     M = len(train_loader)
#     total = 0
#     correct = 0
#     for i, (images, labels) in enumerate((train_loader)):
#         optimizer.zero_grad()
#         labels = labels.to(device)
#         images = images.to(device)
#         if T > 0:
#             outputs = model(images).mean(0)
#         else:
#             outputs = model(images)
#         loss = criterion(outputs, labels)
#         running_loss += loss.item()
#         loss.mean().backward()
#         optimizer.step()
#         total += float(labels.size(0))
#         _, predicted = outputs.cpu().max(1)
#         correct += float(predicted.eq(labels.cpu()).sum().item())
#     return running_loss, 100 * correct / total


# def val(model, test_loader, device, T):
#     correct = 0
#     total = 0
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate((test_loader)):
#             inputs = inputs.to(device)
#             if T > 0:
#                 outputs = model(inputs).mean(0)
#             else:
#                 outputs = model(inputs)
#             _, predicted = outputs.cpu().max(1)
#             total += float(targets.size(0))
#             correct += float(predicted.eq(targets).sum().item())
#         final_acc = 100 * correct / total
#     return final_acc


# utils.py (modified parts only)

def train(model, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0

    total_spikes = 0
    num_inferences = 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if T > 0:
            outputs = model(images)           # expected shape: (T, batch, num_classes)
            # Ensure shape: (T, B, C) -> (B, T, C) for counting
            if outputs.dim() == 3 and outputs.shape[0] == T:
                outputs_perm = outputs.permute(1, 0, 2)  # (B, T, C)
            else:
                # fallback: if model already returns (B, T, C)
                outputs_perm = outputs

            outputs_mean = outputs_perm.mean(1)  # (B, C)
            # count spikes: positive entries across time & classes
            batch_spikes = (outputs_perm > 0).sum().item()
            total_spikes += batch_spikes
            num_inferences += labels.size(0)
            outputs_for_loss = outputs_mean
        else:
            outputs = model(images)  # (B, C)
            outputs_for_loss = outputs

        loss = criterion(outputs_for_loss, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()

        total += float(labels.size(0))
        _, predicted = outputs_for_loss.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

    # compute avg spikes per inference (if any)
    avg_spikes = (total_spikes / num_inferences) if num_inferences > 0 else 0.0
    return running_loss, 100 * correct / total, avg_spikes


def val(model, test_loader, device, T):
    correct = 0
    total = 0
    model.eval()

    total_spikes = 0
    num_inferences = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  # (T, B, C) or (B, C)

            if T > 0:
                # normalize shape to (B, T, C)
                if outputs.dim() == 3 and outputs.shape[0] == T:
                    outputs_perm = outputs.permute(1, 0, 2)
                else:
                    outputs_perm = outputs
                mean_out = outputs_perm.mean(1)  # (B, C)

                # Spike counting
                batch_spikes = (outputs_perm > 0).sum().item()
                total_spikes += batch_spikes
                num_inferences += inputs.size(0)
            else:
                mean_out = outputs

            _, predicted = mean_out.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets.cpu()).sum().item())

    final_acc = 100 * correct / total
    avg_spikes_per_inf = (total_spikes / num_inferences) if num_inferences > 0 else 0.0
    return final_acc, avg_spikes_per_inf
