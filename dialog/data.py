import itertools

import torch
import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from typing import Iterable


def tokenize_context(lines: Iterable[str], tokenizer, eos=True):
    conv = [tokenizer.encode(line) + [tokenizer.eos_token_id] for line in lines]
    return list(itertools.chain.from_iterable(conv))


def to_torch(tokens):
    return torch.tensor(tokens, dtype=torch.long)


def contextualize(lines, num_context):
    return [
        lines[conv_start_index : conv_start_index + num_context]
        for conv_start_index, _ in enumerate(lines[num_context:])
    ]


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, contexted):

        self.examples = []
        for lines in tqdm.tqdm(contexted, desc="Tokenizing"):
            conv = tokenize_context(lines, tokenizer)
            self.examples.append(conv)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return to_torch(self.examples[item])
