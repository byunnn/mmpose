import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet152

class LSTM(nn.Module):
    
    def __init__(self, 
                 mode,
                 window_size,
                 input_type,
                 input_size, 
                 output_size, 
                 hidden_dim, 
                 num_layers, 
                 dropout,
                 bidirectional=False):
        super(LSTM,self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size    
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dense layer to predict 
        self.fc = nn.Linear(hidden_dim, output_size)

        # LSTM layer process the vector sequences 
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_dim,
                            num_layers = self.num_layers,
                            bidirectional = bidirectional,
                            batch_first = True
                           )
        

    def forward(self,x, hidden):
        batch_size = x.size(0) #torch.Size([128, 24, 26])

        #lstm_out : torch.Size([128, 24, 64])
        #hidden[0] : hidden torch.Size([1, 128, 64])
        #hidden[1] : hidden torch.Size([1, 128, 64])
        lstm_out, hidden = self.lstm(x, hidden)  

        #shape : torch.Size([3072, 64])
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        last_output = hidden[0][-1].squeeze(0)

        out = self.dropout(last_output)
        out = self.fc(out)

        return out, hidden

    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden
        
