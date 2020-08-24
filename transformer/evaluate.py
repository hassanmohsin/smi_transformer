# coding: utf-8
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import *
from transformer.build_vocab import WordVocab
from transformer.dataset import Seq2seqDatasetProp
from transformer.models import TrfmSeq2seqProp2

torch.manual_seed(0)

if len(sys.argv) != 4:
    print("Usage: python -m transformer.evaluate <param_file> <test_data> <model_file>")
    sys.exit()

PARAMS = sys.argv[1]
TESTDATA = sys.argv[2]
MODELFILE = sys.argv[3]

assert torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = json.load(open(PARAMS))
vocab = WordVocab.load_vocab(params['vocab'])

model = TrfmSeq2seqProp2(len(vocab), params['hidden'], len(vocab), params['n_layer'])
model.load_state_dict(torch.load(MODELFILE))
model = nn.DataParallel(model)
model.to(device)
model.eval()

# Load the scaler
scaler = joblib.load("data/scaler.pkl")


def get_metrics():
    # Evaluate test data
    batch_size = params['batch_size']
    dataset = Seq2seqDatasetProp(params, vocab)

    _, test = torch.utils.data.random_split(
        dataset, [len(dataset) - params['test_size'], params['test_size']])

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=56)
    ytrue = np.empty((len(test), 3), np.float32)
    ypred = np.empty_like(ytrue, np.float32)

    print(f"Evaluating {params['data']}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            _, pred = model(x.to(device))
            ytrue[i * batch_size:i * batch_size + batch_size] = y.numpy()
            ypred[i * batch_size:i * batch_size + batch_size] = pred.cpu().numpy()

    # "logP", "qed", "sas"
    # Inverse transform
    ytrue = scaler.inverse_transform(ytrue)
    ypred = scaler.inverse_transform(ypred)
    mae = [mean_absolute_error(ytrue[:, i], ypred[:, i]) for i in range(3)]
    rmse = [np.sqrt(mean_squared_error(ytrue[:, i], ypred[:, i])) for i in range(3)]

    print(f"VALUES: {params['props']}")
    print(f"MEAN ABSOLUTE ERROR: {mae}")
    print(f"ROOT MEAN SQUARED ERROR: {rmse}")

    print("--------------------------------")

    # Evaluate zinc data
    params['data'] = TESTDATA  # replace the training data file with the test data file
    dataset = Seq2seqDatasetProp(params, vocab)
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=56, shuffle=False)
    ytrue = np.empty((len(dataset), 3), np.float32)
    ypred = np.empty_like(ytrue, np.float32)

    print(f"Evaluating {params['data']}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            _, pred = model(x.to(device))
            ytrue[i * batch_size:i * batch_size + batch_size] = y.numpy()
            ypred[i * batch_size:i * batch_size + batch_size] = pred.cpu().numpy()

    # "logP", "qed", "sas"
    # Inverse transform
    ytrue = scaler.inverse_transform(ytrue)
    ypred = scaler.inverse_transform(ypred)
    mae = [mean_absolute_error(ytrue[:, i], ypred[:, i]) for i in range(3)]
    rmse = [np.sqrt(mean_squared_error(ytrue[:, i], ypred[:, i])) for i in range(3)]
    print(f"VALUES: {params['props']}")
    print(f"MEAN ABSOLUTE ERROR: {mae}")
    print(f"ROOT MEAN SQUARED ERROR: {rmse}")


def encode():
    output = np.empty((len(dataset), 1024), dtype=np.float32)
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            _encoder = model.module._encode(x.to(device))
            output[i * batch_size:i * batch_size + batch_size] = _encoder
    np.save('encoded', output)
    print("Encoded representations are saved to encoded.npy")


if __name__ == "__main__":
    get_metrics()
    # encode()
