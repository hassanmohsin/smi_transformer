# coding: utf-8
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from tqdm import *
import numpy as np
from transformer.models import TrfmSeq2seqProp2
from transformer.dataset import Seq2seqDatasetProp
from transformer.build_vocab import WordVocab
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(0)

assert torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

params = json.load(open("exp_params/params_4.json"))
params['data'] = "data/250k_rndm_zinc_drugs_clean_3_scaled.csv"
vocab = WordVocab.load_vocab('data/vocab.pkl')
test_size = params['test_size']
dataset = Seq2seqDatasetProp(params, vocab)
batch_size = 1024
test_loader = DataLoader(dataset, batch_size = batch_size, num_workers=56, shuffle=False)


model = TrfmSeq2seqProp2(len(vocab), params['hidden'], len(vocab), params['n_layer'])
model.load_state_dict(torch.load("exps/exp4/ST_48_1767-0.003617.pkl"))
model = nn.DataParallel(model)
model.to(device)
model.eval()

ytrue = np.empty((len(dataset), 3), np.float32)
ypred = np.empty_like(ytrue, np.float32)

print(f"Evaluating {params['data']}...")
with torch.no_grad():
    for i, (x, y) in enumerate(tqdm(test_loader)):
        _, pred = model(x.to(device))
        for p, q in zip(y, pred):
            ytrue[i*batch_size:i*batch_size+batch_size] = p.cpu().numpy()
            ypred[i*batch_size:i*batch_size+batch_size] = q.cpu().numpy()
    
#"logP", "qed", "sas"
mae = [mean_absolute_error(ytrue[:, i], ypred[:, i]) for i in range(3)]
rmse = [np.sqrt(mean_squared_error(ytrue[:, i], ypred[:, i])) for i in range(3)]
print(f"VALUES: {params['props']}")
print(f"MEAN ABSOLUTE ERROR: {mae}")
print(f"ROOT MEAN SQUARED ERROR: {rmse}")

