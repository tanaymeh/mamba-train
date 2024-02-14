import numpy as np

import torch
from torch.utils import Dataset

import lance

class MambaDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        context_len,
        fim_prefix,
        fim_middle,
        fim_suffix,
        fim_pad,
        fim_rate=0.5,
        rng_seed=42
    ):
        # Load the lance dataset from the saved path
        self.ds = lance.dataset(dataset_path)
        self.context_len = context_len
        # Doing this so the sampler never asks for an index at the end of text
        self.length = self.ds.count_rows() - context_len
        
        self.np_rng = np.random.RandomState(seed=rng_seed)

        self.fim_prefix = fim_prefix
        self.fim_middle = fim_middle
        self.fim_suffix = fim_suffix
        self.fim_pad = fim_pad
        self.fim_rate = fim_rate

    def __len__(self):
        return self.length

    def from_idxs(self, idxs):
        """
        Little utility function to get the data from lance
        """
        data = self.ds.take(idxs).to_pylist()
        data = torch.tensor(list(map(lambda x: x['value'], data)))
        return data

    def apply_fim(self, sample):
        """
        Applies FIM transformation on one sample
        """
        boundaries = sorted(self.np_rng.randint(low=0, high=len(sample)+1, size=2))

        prefix = sample[: boundaries[0]]
        middle = sample[boundaries[0] : boundaries[1]]
        suffix = sample[boundaries[1] :]

        total_length = len(prefix) + len(middle) + len(suffix) + 3
        diff = total_length - len(sample)
        if diff > 0:
            suffix = suffix[: max(0, len(suffix) - diff)]
        elif diff < 0:
            suffix.extend([self.fim_pad] * (-diff))
        
        if self.np_rng.binomial(1, self.fim_rate):
            # Apply SPM
            transfomed_example = [self.fim_prefix, self.fim_suffix] + suffix + [self.fim_middle] + prefix + middle
        else:
            # Apply PSM
            transfomed_example = [self.fim_prefix] + prefix + [self.fim_suffix] + suffix + [self.fim_middle] + middle
        
        return transfomed_example

    def __getitem__(self, idx):
        """
        Generate a list of indices starting from the current idx to idx+context_len+1
        Use the from_idxs function to get data in said indexes and then divide it into features (x) and target (y)
        """
        current_window_idxs = np.arange(idx, idx+self.context_len+1)
        sample = self.from_idxs(current_window_idxs)

        # Apply FIM transformation depending on the rate
        if self.np_rng.binomial(1, self.fim_rate):
            sample = self.apply_fim(sample)

        tokens = sample[0:self.context_len]
        labels = sample[1:self.context_len+1] # +1 because our target is the sentence is 1 step ahead of input text
        return tokens, labels