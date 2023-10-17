import json
import torch.utils as utils
from transformers import T5TokenizerFast as Tokenizer
from detector.t5_hidden.__init__ import config
from torch import Tensor
from typing import Tuple


class Dataset(utils.data.Dataset):
    '''
    Dataset for loading text from different large language models.

    Attributes:
        corpus (list[str]): The corpus of the dataset.
        label (list[str]): The labels of the dataset.
        tokenizer (Tokenizer): The tokenizer used.
    '''
    def __init__(self, partition: str, selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'PaLM', 'LLaMA', 'GPT2')):
        super().__init__()
        
        self.corpus, self.label = [], []
        filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        for item in filteredDataset:
            with open(f'{item.root}/{partition}.jsonl', 'r') as f:
                for line in f:

                    if item.label == 'LLaMA':
                        words = json.loads(line)['text'].split()
                        continuation = words[75:]
                        if len(continuation) >= 42:
                            self.corpus.append(' '.join(continuation[:256]))
                            self.label.append(item.token)
                    else:
                        self.corpus.append(json.loads(line)['text'])
                        self.label.append(item.token)
                    
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(config.backbone.name, model_max_length=config.backbone.model_max_length)
        
    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.corpus[idx], self.label[idx]

    def collate_fn(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor, Tensor]:
        corpus, label = zip(*batch)
        corpus = self.tokenizer.batch_encode_plus(corpus, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        label = self.tokenizer.batch_encode_plus(label, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        return corpus.input_ids, corpus.attention_mask, label.input_ids
