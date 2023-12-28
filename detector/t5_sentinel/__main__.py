import os
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from detector.t5_sentinel.dataset import Dataset
from detector.t5_sentinel.model import Sentinel
from detector.t5_sentinel.utilities import train, validate
from detector.t5_sentinel.__init__ import config


##############################################################################
# Dataset and Dataloader
##############################################################################
train_loader = DataLoader(
    train_dataset := Dataset("train-dirty"),
    collate_fn=train_dataset.collate_fn,
    batch_size=config.dataloader.batch_size,
    num_workers=config.dataloader.num_workers,
    shuffle=True,
)


valid_loader = DataLoader(
    valid_dataset := Dataset("valid-dirty"),
    collate_fn=valid_dataset.collate_fn,
    batch_size=config.dataloader.batch_size,
    num_workers=config.dataloader.num_workers,
    shuffle=False,
)


##############################################################################
# Model, Optimizer, and Scheduler
##############################################################################
model = Sentinel().cuda()

if cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.AdamW(
    model.parameters(),
    lr=config.optimizer.lr,
    weight_decay=config.optimizer.weight_decay,
)

##############################################################################
# Task and Cache
##############################################################################

task = wandb.init(
    name=config.id,
    project="llm-sentinel",
    entity="deep-learner",
    id="5gk3khsd",
    resume="must",
)

wandb.save("detector/t5_sentinel/__init__.py")
wandb.save("detector/t5_sentinel/__main__.py")
wandb.save("detector/t5_sentinel/dataset.py")
wandb.save("detector/t5_sentinel/model.py")
wandb.save("detector/t5_sentinel/settings.yaml")
wandb.save("detector/t5_sentinel/types.py")
wandb.save("detector/t5_sentinel/utilities.py")

cache = f"storage/{config.id}"
os.path.exists(cache) or os.makedirs(cache)

if os.path.exists(f"{cache}/state.pt"):
    state = torch.load(f"{cache}/state.pt")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    startEpoch = state["epochIter"] + 1
    bestValidationAccuracy = state["validAccuracy"]
else:
    startEpoch = 0
    bestValidationAccuracy = float("-inf")


##############################################################################
# Training and Validation
##############################################################################
for epoch in range(startEpoch, config.epochs):
    tqdm.write("Epoch {}".format(epoch + 1))
    learnRate = optimizer.param_groups[0]["lr"]
    trainLoss, trainAccuracy = train(model, optimizer, train_loader)
    validAccuracy = validate(model, valid_loader)

    wandb.log(
        {
            "Training Loss": trainLoss,
            "Training Accuracy": trainAccuracy * 100,
            "Validation Accuracy": validAccuracy * 100,
            "Learning Rate": learnRate,
        }
    )

    tqdm.write("Training Accuracy {:.2%}".format(trainAccuracy))
    tqdm.write("Training Loss {:.4f}".format(trainLoss))
    tqdm.write("Validation Accuracy {:.2%}".format(validAccuracy))
    tqdm.write("Learning Rate {:.4f}".format(learnRate))

    checkpoint = {
        "epochIter": epoch,
        "model": model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validAccuracy": validAccuracy,
    }

    if validAccuracy > bestValidationAccuracy:
        bestValidationAccuracy = validAccuracy
        torch.save(checkpoint, f"{cache}/state.pt")
        tqdm.write("Checkpoint Saved!")
