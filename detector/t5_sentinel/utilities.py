import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from detector.t5_sentinel.__init__ import config
from detector.t5_sentinel.types import SentinelOutput
from typing import Tuple


def train(
    model: nn.Module,
    optimizer: nn.Module,
    dataloader: DataLoader,
    selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'PaLM', 'LLaMA')
) -> Tuple[float, float]:
    model.train()
    accumulatedLoss, accumulatedCorrect, accumulatedBatchSize = 0, 0, 0
    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', ncols=120)

    filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
    
    for i, (corpus_ids, corpus_mask, label_ids) in progress:
        
        output: SentinelOutput = model.forward(corpus_ids.cuda(), corpus_mask.cuda(), label_ids.cuda())
        loss, probabilities, predictions = output.huggingface.loss, output.probabilities, []
        for argmaxIndex in probabilities.argmax(dim=-1):
            predictions.append(filteredDataset[argmaxIndex].token_id)
        
        accumulatedLoss += loss.mean().item()
        accumulatedCorrect += sum([1 if prediction == label_id[0] else 0 for prediction, label_id in zip(predictions, label_ids.tolist())])
        accumulatedBatchSize += config.dataloader.batch_size

        loss.mean().backward()
        if accumulatedBatchSize >= config.optimizer.batch_size or i == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            accumulatedBatchSize = 0

        progress.set_postfix({
            'loss': '{:04f}'.format(accumulatedLoss / (i + 1)), 
            'accuracy': '{:04%}'.format(accumulatedCorrect / ((i + 1) * config.dataloader.batch_size))
        })
    
    progress.close()
    return accumulatedLoss / len(dataloader), accumulatedCorrect / (len(dataloader) * config.dataloader.batch_size)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'PaLM', 'LLaMA')
) -> float:
    model.eval()
    accumulatedCorrect = 0
    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating', ncols=120)

    filteredDataset = [item for item in config.dataset if item.label in selectedDataset]

    for i, (corpus_ids, corpus_mask, label_ids) in progress:
        output: SentinelOutput = model.forward(corpus_ids.cuda(), corpus_mask.cuda(), label_ids.cuda())
        probabilities, predictions = output.probabilities, []
        for argmaxIndex in probabilities.argmax(dim=-1):
            predictions.append(filteredDataset[argmaxIndex].token_id)
        
        accumulatedCorrect += sum([1 if prediction == label_id[0] else 0 for prediction, label_id in zip(predictions, label_ids.tolist())])
        progress.set_postfix({
            'accuracy': '{:04%}'.format(accumulatedCorrect / ((i + 1) * config.dataloader.batch_size))
        })
    
    progress.close()
    return accumulatedCorrect / (len(dataloader) * config.dataloader.batch_size)
