import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, output_size, hidden_dim, num_layers, embedding_dim, seq_len, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.device = device

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=0.35,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input):
        seq_lengths = input[:, -1, 0]
        pack = pack_padded_sequence(input, seq_lengths.clamp(max=self.seq_len).to('cpu'),
                                    enforce_sorted=False, batch_first=True).to(self.device)

        h0 = torch.ones(self.num_layers * 2, input.size(0),
                        self.hidden_dim).to(self.device)
        c0 = torch.ones(self.num_layers * 2, input.size(0),
                        self.hidden_dim).to(self.device)

        _, (ht, ct) = self.lstm(pack, (h0, c0))
        out = self.fc1(ht[-1])
        out = self.fc(out)

        return out
