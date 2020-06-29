from .build_vocab import WordVocab
import pandas as pd
import hiddenlayer as hl
from torchsummaryX import summary
from .models import TrfmSeq2seqProp2
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--vocab", "-v", default='data/vocab.pkl',
                        type=str,
                        help="vocabulary (.pkl)")
    parser.add_argument("--seq_len", "-len", type=int,
                        default=220, help="Sequence length")
    parser.add_argument("--hidden",
                        type=int,
                        default=256,
                        help="length of hidden vector")
    parser.add_argument("--n_layer",
                        "-l",
                        type=int,
                        default=4,
                        help="number of layers")
    parser.add_argument("--batch_size", "-b", type=int,
                        default=8, help="Batch size.")
    args = parser.parse_args()

    vocab = WordVocab.load_vocab(args.vocab)

    model = TrfmSeq2seqProp2(len(vocab), args.hidden,
                             len(vocab), args.n_layer)
    print(model)
    print(summary(model, torch.zeros(args.batch_size, args.seq_len).long()))
    print(len(vocab)) 
