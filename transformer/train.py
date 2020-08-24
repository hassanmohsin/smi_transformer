import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchsummaryX import summary
from tqdm import tqdm

from .build_vocab import WordVocab
from .dataset import Seq2seqDatasetProp
from .hyperparameters import load_params
from .models import TrfmSeq2seqProp2

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
torch.manual_seed(0)


def evaluate(params, model, test_loader, vocab):
    model.eval()
    total_loss = 0
    total_reproduction_loss = 0
    total_pred_loss = 0
    for b, (sm, prop) in enumerate(test_loader):
        sm = sm.cuda()  # (T,B)
        # sm = torch.t(sm.cuda()) # (T,B)
        prop = prop.cuda()
        with torch.no_grad():
            output, ppred = model(sm)  # (T,B,V)
        reproduction_loss = F.nll_loss(
            # output.view(-1, len(vocab)), sm.contiguous().view(-1), ignore_index=PAD
            output.view(-1, len(vocab)),
            sm.contiguous().view(-1))

        # output, _ = output.max(axis=2)
        # reproduction_loss = F.kl_div(output, sm.contiguous().float())
        pred_loss = (F.mse_loss(prop[:, 0], ppred[:, 0]) +
                     F.mse_loss(prop[:, 1], ppred[:, 1]) +
                     F.mse_loss(prop[:, 2], ppred[:, 2]))
        loss = params["rep_loss_weight"] * reproduction_loss + \
               params["pred_loss_weight"] * pred_loss
        total_reproduction_loss += reproduction_loss.item()
        total_pred_loss += pred_loss.item()
        total_loss += loss.item()
    # return total_loss / len(test_loader)
    return (
        total_reproduction_loss / len(test_loader),
        total_pred_loss / len(test_loader),
        total_loss / len(test_loader),
    )


def main():
    parser = argparse.ArgumentParser(description="Train the transformer model")
    parser.add_argument("--exp_dir", "-e", type=str,
                        required=True, help="Experiment directory")
    parser.add_argument("--batch_size", "-b",
                        type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--n_worker",
                        type=int,
                        default=16,
                        help="number of workers")
    parser.add_argument("--gpu",
                        metavar="N",
                        type=int,
                        nargs="+",
                        help="list of GPU IDs to use")
    args = vars(parser.parse_args())
    json_file = os.path.join(args['exp_dir'], 'params.json')
    assert os.path.isfile(json_file)
    params = load_params(json_file)
    params.update(args)
    print(f"Training parameters: {params}")

    assert torch.cuda.is_available()

    print("[INFO] Loading dataset...")
    vocab = WordVocab.load_vocab(params["vocab"])
    test_idx = np.load("data/test_idx.npy")
    dataset = Seq2seqDatasetProp(params, vocab)
    train_idx = np.delete(np.arange(len(dataset)), test_idx)
    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    print(f"Training data: {len(train)}, Test data: {len(test)}")
    # test_size = params["test_size"]
    # train, test = torch.utils.data.random_split(
    #    dataset, [len(dataset) - test_size, test_size])
    train_loader = DataLoader(train,
                              batch_size=params["batch_size"],
                              shuffle=True,
                              num_workers=params["n_worker"])  # , drop_last=True)
    test_loader = DataLoader(test,
                             batch_size=params["batch_size"],
                             shuffle=False,
                             num_workers=params["n_worker"])  # , drop_last=True)
    print("[INFO] Train size:", len(train))
    print("[INFO] Test size:", len(test))
    del dataset, train, test
    # for b, (sm, prop) in tqdm(enumerate(train_loader)):
    #     print(sm.size(), prop.size())
    #     # break

    model = TrfmSeq2seqProp2(len(vocab), params["hidden"], len(vocab),
                             params["n_layer"]).cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    print("[INFO] Total parameters:", sum(p.numel()
                                          for p in model.parameters()))

    best_loss = None
    losses = []
    print('[INFO] Training the model...')
    for e in range(1, params["epochs"]):
        for b, (sm, prop) in tqdm(enumerate(train_loader)):
            sm = sm.cuda()  # (T,B)
            prop = prop.cuda()  # logP, qed and sas values (dataset class is calling in this order)
            # print(sm.size(), prop.size())
            optimizer.zero_grad()
            output, ppred = model(sm)  # (T,B,V)
            reproduction_loss = F.nll_loss(
                # output.view(-1, len(vocab)), sm.contiguous().view(-1), ignore_index=PAD
                output.view(-1, len(vocab)),
                sm.contiguous().view(-1))
            # reproduction_loss = F.kl_div(sm.contiguous().float(), output, reduction='none').mean(dim=(2))

            # print(sm.size(), output.size(), ppred.size())
            # print(f"{output.view(-1, len(vocab)).size()}, {sm.contiguous().view(-1, len(vocab)).size()}")
            # print(
            #     output.view(-1, len(vocab)).type(),
            #     sm.contiguous().view(-1).float().type(),
            # )
            # output, _ = output.max(axis=2)
            pred_loss = (F.mse_loss(prop[:, 0], ppred[:, 0]) +
                         F.mse_loss(prop[:, 1], ppred[:, 1]) +
                         F.mse_loss(prop[:, 2], ppred[:, 2]))
            loss = params["rep_loss_weight"] * reproduction_loss + \
                   params["pred_loss_weight"] * pred_loss

            loss.backward()
            optimizer.step()
            if b % 100 == 0:
                print(
                    "Train {:3d}: iter {:5d} | pred. loss {:.3f} | rep. loss {:.3f} | total loss {:.3f} | ppl {:.3f}"
                        .format(
                        e,
                        b,
                        pred_loss.item(),
                        reproduction_loss.item(),
                        loss.item(),
                        math.exp(reproduction_loss.item()),
                    ))

        eval_rep_loss, eval_pred_loss, eval_total_loss = evaluate(params,
                                                                  model, test_loader, vocab)
        print(
            "Val {:3d}: iter {:5d} | pred. loss {:.3f} | rep. loss {:.3f} | total loss {:.3f} | ppl {:.3f}"
                .format(
                e,
                b,
                eval_pred_loss,
                eval_rep_loss,
                eval_total_loss,
                math.exp(eval_rep_loss),
            ))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_loss or eval_total_loss < best_loss:
            print("[INFO] Saving model...")
            if not os.path.isdir(params["exp_dir"]):
                os.makedirs(params["exp_dir"])
            torch.save(
                model.module.state_dict(),
                "./%s/%s_%d_%d-%f.pkl" %
                (params["exp_dir"], params["name"], e, b, eval_total_loss),
            )
            best_loss = eval_total_loss

        # Log the losses
        losses.append([reproduction_loss.item(), pred_loss.item(), loss.item(),
                       eval_rep_loss, eval_pred_loss, eval_total_loss])
        # scheduler.step()

    pd.DataFrame.from_records(losses,
                              columns=[
                                  'train_rep_loss',
                                  'train_pred_loss',
                                  'train_loss',
                                  'eval_rep_loss',
                                  'eval_pred_loss',
                                  'eval_loss'
                              ]).to_csv(
        os.path.join(params["exp_dir"], params["history"]), index=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
