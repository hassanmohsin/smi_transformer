from build_vocab import WordVocab
from dataset import Seq2seqDataset
import pandas as pd
import hiddenlayer as hl
from torchsummaryX import summary
from pretrain_trfm import TrfmSeq2seq
import torch

vocab = WordVocab.load_vocab("../data/vocab.pkl")
dataset = Seq2seqDataset(pd.read_csv("../data/chembl_24.csv")['canonical_smiles'].values, vocab)
model = TrfmSeq2seq(len(vocab), 220, len(vocab), 3)
print(model)
print(summary(model, torch.zeros(220, 8).long()))
