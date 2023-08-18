import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet152

class LSTM(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_dim, 
                 num_layers, 
                 bidirectional=False):
        super(LSTM,self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size    
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dense layer to predict 
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.dropout = nn.Dropout(dropout)
        # Prediction activation function
        self.softmax = nn.Softmax(dim=1)

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

        last_output = hidden[0].squeeze(0)
        out = self.fc(last_output)
        out = self.softmax(out)

        return out, hidden

        # # out = self.dropout(lstm_out)

        # #out : torch.Size([3072, 20])
        # out = self.fc(lstm_out)
  
        # #softmax :  torch.Size([3072, 20])
        # out = self.softmax(out)

        # #마지막 LSTM cell의 output만 추출
        # out = out.view(batch_size, -1)  # torch.Size([128, 480])
        # out = out[:, -1] #torch.Size([128])
        # return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden
        
        # packed_output,(hidden_state,cell_state) = self.lstm(x)
        
        # # Concatenating the final forward and backward hidden states
        # hidden = self.dropout(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        #LSTM은 output을 어떻게 출력하냐에 따라 분류, 생성 모델로 활용 가능
        #분류의 경우 마지막 hideen layer를 추출해서 classifier를 연결해주면 사용이 가능하다.
        # # 방법 1 (마지막 LSTM cell의 hidden_state 추출)
        # last_output = hidden[0].squeeze()
        # print(last_output)
        # # 방법 2 (out의 마지막 열들을 모두 추출)
        # last_output = out[:,-1]
        # print(last_output)



class CNN(torch.nn.Module):

    def __init__(self, layer):
        super(CNN, self).__init__()

        self.layer = layer 

        if self.layer =='keypoint' or self.layer == 'skeleton_sub' :    
            
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU())  

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU())

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc = torch.nn.Linear(128 * 6 * 1, 20, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)


        elif layer == 'skeleton_matrix' :
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU())
            
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU())

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc = torch.nn.Linear(128 * 6 * 2, 20, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)


        elif layer == 'sequence' :
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU())
            
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU())
            
            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc = torch.nn.Linear(256 * 3 * 12, 20, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)

        else :
            print("model input type error")


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.layer =='keypoint' or self.layer == 'skeleton_sub' :    
            out = out.view(-1, 128 * 6)

        elif self.layer == 'skeleton_matrix' :
            out = out.view(-1, 128 * 6 * 2)

        elif self.layer == 'sequence' :
            out = self.layer4(out)
            out = out.view(-1, 256 * 3 * 12)
            
        out = self.fc(out)

        return out