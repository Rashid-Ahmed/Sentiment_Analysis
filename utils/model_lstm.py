import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


#LSTM model we use for sentiment analysis
class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_dim, num_layers, embedding_dim, seq_len, device):
        
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.device = device
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, dropout = 0.35, num_layers = num_layers, batch_first=True, bidirectional = True)
        #batch_first = True means batch has dimension 0
        # we are using bidirectional = True, meaning the model starts looking at a sentence from both word 0 to word N and word N to word 0
        #input sequence to classifier is  (batchsize, padded sequence length, embedding size)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        #self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        
        seq_lengths = input[: , -1 , 0]
        pack = pack_padded_sequence(input, seq_lengths.clamp(max=self.seq_len).to('cpu'), enforce_sorted = False, batch_first=True).to(self.device)
        # pack padded sequence helps us with efficient computation by not using sparse values (0 values we have for some sentences that have length < sequence length  are sparse values)
        h0 = torch.ones(self.num_layers * 2, input.size(0), self.hidden_dim).to(self.device)
        #h0 is the first hidden state(at time step 0) for the lstm network and input.size(0) is our batch size
        c0 = torch.ones(self.num_layers * 2, input.size(0), self.hidden_dim).to(self.device)
        #c0 is the first cell state(at time step 0) for the lstm network (lstms have cells as well unlike gru or rnn which is why we need this c0)
   
        hidden, (ht, ct) = self.lstm(pack, (h0, c0))
        #out is of shape(batch_size, seq_length, hidden_size))
        # batch_size means the output for each sentence in a batch
        # seq_length means that for each time step we get a output(i.e for each word we get a output) however we only need the last output after the last word
        # hidden_size this is for the output features of the LSTM layers we feed this to our linear layer to get our number of outputs
        out = self.fc1(ht[-1])
        out = self.fc(out)

        # this is how we take only the last time steps features for all the batches.
        return out
		    


    