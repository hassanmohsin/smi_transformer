import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .build_vocab import WordVocab
from .dataset import Seq2seqDatasetProp
from .models import *

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--data', '-d', type=str, default='data/chembl_25.csv', help='train corpus (.csv)')
    parser.add_argument('--out_dir', '-o', type=str, default='output', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


def evaluate(model, test_loader, vocab):
    model.eval()
    total_loss = 0
    total_reproduction_loss = 0
    total_pred_loss = 0
    for b, (sm, prop) in enumerate(test_loader):
        sm = sm.cuda() # (T,B)
        #sm = torch.t(sm.cuda()) # (T,B)
        prop = prop.cuda()
        with torch.no_grad():
            output, ppred = model(sm) # (T,B,V)
        reproduction_loss = F.nll_loss(output.view(-1, len(vocab)),
                                       sm.contiguous().view(-1),
                                       ignore_index=PAD)
        pred_loss = F.mse_loss(prop[:,0], ppred[:, 0]) + F.mse_loss(prop[:,1], ppred[:, 1]) + F.mse_loss(prop[:,2], ppred[:, 2])
        loss = reproduction_loss + pred_loss
        #loss = 0.9*reproduction_loss + 0.1*pred_loss
        total_reproduction_loss += reproduction_loss.item()
        total_pred_loss += pred_loss.item()
        total_loss += loss.item()
    #return total_loss / len(test_loader)
    return total_reproduction_loss / len(test_loader), total_pred_loss / len(test_loader), total_loss/len(test_loader)

def main():
    args = parse_arguments()
    assert torch.cuda.is_available()

    print('Loading dataset...')
    vocab = WordVocab.load_vocab(args.vocab)
    dataset = Seq2seqDatasetProp(pd.read_csv(args.data), vocab)
    test_size = 10000
    train, test = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    print('Train size:', len(train))
    print('Test size:', len(test))
    del dataset, train, test

    model = TrfmSeq2seqProp2(len(vocab), args.hidden, len(vocab), args.n_layer).cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))


    best_loss = None
    for e in range(0, args.n_epoch):
        for b, (sm, prop) in tqdm(enumerate(train_loader)):
            sm = sm.cuda() # (T,B)
            prop = prop.cuda()
            optimizer.zero_grad()
            output, ppred = model(sm) # (T,B,V)
            reproduction_loss = F.nll_loss(output.view(-1, len(vocab)),
                                sm.contiguous().view(-1), ignore_index=PAD)
            pred_loss = F.mse_loss(prop[:,0], ppred[:, 0]) + \
                        F.mse_loss(prop[:,1], ppred[:, 1]) + \
                        F.mse_loss(prop[:,2], ppred[:, 2])
            loss = reproduction_loss + pred_loss
            #loss = 0.9*reproduction_loss + 0.1*pred_loss
            loss.backward()
            optimizer.step()
            if b%50==0:
                print('Train {:3d}: iter {:5d} | pred. loss {:.3f} | rep. loss {:.3f} | total loss {:.3f} | ppl {:.3f}'.format(e, b, pred_loss.item(), reproduction_loss.item(), loss.item(), math.exp(reproduction_loss.item())))
            #if b%10000==0:
        eval_rep_loss, eval_pred_loss, eval_total_loss  = evaluate(model, test_loader, vocab)
        print('Val {:3d}: iter {:5d} | pred. loss {:.3f} | rep. loss {:.3f} | total loss {:.3f} | ppl {:.3f}'.format(e, b, eval_pred_loss, eval_rep_loss, eval_total_loss, math.exp(eval_rep_loss)))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_loss or eval_total_loss < best_loss:
            print("[!] saving model...")
            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            torch.save(model.module.state_dict(), './%s/%s_%d_%d-%f.pkl' % (args.out_dir,args.name,e,b,eval_total_loss))
            best_loss = eval_total_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
        
