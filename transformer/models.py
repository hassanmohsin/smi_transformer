import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import numpy as np

__all__ = ["TrfmSeq2seq", "TrfmSeq2seqProp", "TrfmSeq2seqProp2"]


class PredictorModel(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.3):
        super(PredictorModel, self).__init__()
        self.linear1 = nn.Linear(in_size, 1024)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.linear5 = nn.Linear(16, out_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.relu(self.bn1(self.linear3(x)))
        x = F.relu(self.bn2(self.linear4(x)))
        x = self.linear5(x)

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (T,H)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        hidden = self.trfm(embedded, embedded)  # (T,B,H)
        out = self.out(hidden)  # (T,B,V)
        out = F.log_softmax(out, dim=2)  # (T,B,V)
        return out  # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack(
            [
                np.mean(output, axis=0),
                np.max(output, axis=0),
                output[0, :, :],
                penul[0, :, :],
            ]
        )  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:  # Batch is too large to load
            print(
                "There are {:d} molecules. It will take a little time.".format(
                    batch_size
                )
            )
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out


class TrfmSeq2seqProp(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seqProp, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.predict = PredictorModel(220 * hidden_size, 3)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        # src = src.transpose(0, 1)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        hidden = self.trfm(embedded, embedded)  # (T,B,H)
        flatten = torch.flatten(hidden, start_dim=1)  # (B, 56320)
        # flatten = torch.flatten(hidden.transpose(0, 1), start_dim=1) # (B, 56320)
        pred = self.predict(flatten)
        out = self.out(hidden)  # (T,B,V)
        out = F.log_softmax(out, dim=2)  # (T,B,V)
        return out, pred  # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack(
            [
                np.mean(output, axis=0),
                np.max(output, axis=0),
                output[0, :, :],
                penul[0, :, :],
            ]
        )  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:  # Batch is too large to load
            print(
                "There are {:d} molecules. It will take a little time.".format(
                    batch_size
                )
            )
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out


class TrfmSeq2seqProp2(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seqProp2, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.predict = PredictorModel(hidden_size, 3)
        # self.predict = PredictorModel(220*hidden_size, 3)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (B,T)
        # src = src.transpose(0, 1)
        # print('src', src.size())
        embedded = self.embed(src)  # (B,T,H)
        # print('embd', embedded.size())
        embedded = self.pe(embedded)  # (B,T,H)
        # print('embd', embedded.size())
        output = embedded
        for mod in self.trfm.encoder.layers:
            output = mod(output, None)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)
        encoded = output
        # print('encoded', encoded.size())
        for mod in self.trfm.decoder.layers:
            output = mod(output, embedded)
        if self.trfm.decoder.norm:
            output = self.trfm.decoder.norm(output)
        # print('output ', output.size())
        out = self.out(output)  # (B,T,V)
        # print('out ', out.size())
        out = F.log_softmax(out, dim=2)  # (B,T,V)

        flatten = encoded.mean(1)
        # flatten = torch.flatten(encoded, start_dim=1) # (B, 56320) ##########
        # flatten = torch.flatten(hidden.transpose(0, 1), start_dim=1) # (B, 56320)
        # print('flatten ', flatten.size())
        pred = self.predict(flatten)  # (B,3)
        return out, pred

    def _encode(self, src):
        # src: (B,T)
        embedded = self.embed(src)  # (B,T,H)
        embedded = self.pe(embedded)  # (B,T,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (B,T,H)
        penul = output.cpu().detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (B,T,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (B,T,H)
        output = output.cpu().detach().numpy()
        # mean, max, first*2
        return np.hstack(
            [
                np.mean(output, axis=1),
                np.max(output, axis=1),
                output[:, 0, :],
                penul[:, 0, :],
            ]
        )  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[0]
        if batch_size <= 100:
            return self._encode(src)
        else:  # Batch is too large to load
            print(
                "There are {:d} molecules. It will take a little time.".format(
                    batch_size
                )
            )
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out
