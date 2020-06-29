import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .enumerator import SmilesEnumerator
from .utils import split

PAD = 0


class Randomizer(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.sme = SmilesEnumerator()

    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm)  # Random transoform
        if sm_r is None:
            sm_spaced = split(sm)  # Spacing
        else:
            sm_spaced = split(sm_r)  # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split) <= self.seq_len - 2:
            return sm_split  # List
        else:
            # return split(sm).split()
            return sm_split[:self.seq_len-2]

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)


class Seq2seqDataset(Dataset):
    def __init__(self, smiles, vocab, seq_len=220, transform=True):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = Randomizer(seq_len=self.seq_len)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        sm = self.transform(sm)  # List
        content = [
            self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm
        ]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)


class Seq2seqDatasetProp(Dataset):
    def __init__(self, df, vocab, seq_len=220, transform=True):
        self.smiles = df['canonical_smiles'].values
        self.props = df[['logP', 'qed', 'sas']].values
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = Randomizer(seq_len=self.seq_len)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        prop = self.props[item].tolist()
        if self.transform is not None:
            sm = self.transform(sm)  # List
        content = [
            self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm
        ]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)
        assert len(
            X) == self.seq_len, f"Invalid length of X, {len(X)} instead of {self.seq_len}. Content length is {len(content)}"
        assert len(prop) == 3, f"Invalid length of Y, {len(prop)} instead of 3"
        return torch.tensor(X), torch.tensor(prop)
