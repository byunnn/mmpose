import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet152


class CNN_multiframe(torch.nn.Module):

    def __init__(self, mode, dropout, input_type, n_layer, window_size=24 ):
        super(CNN_multiframe, self).__init__()

        self.input_type = input_type 
        self.window_size = window_size
        self.dropout = dropout
        self.n_layer = n_layer

        if input_type == 'skeleton_sub' and window_size == 24:

            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout))
            
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout))
            
            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(256 * 3 * 12, 20, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)


        elif input_type == 'skeleton_sub' and window_size == 15 and n_layer == 2:

            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(64 * 3 * 7, 6, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)



        elif input_type == 'skeleton_sub' and window_size == 9 and n_layer == 2:

            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(64 * 3 * 4, 6, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)

            
        elif input_type == 'skeleton_sub' and window_size == 9 and n_layer == 4:
            
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout))
            
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout))
            
            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(256 * 3 * 4, 6, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)


        elif input_type == 'skeleton_angle' and window_size == 15 and n_layer == 2:
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(64 * 3 * 11, 6, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)

            
        elif input_type == 'skeleton_angle' and window_size == 9 and n_layer == 2:
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(64 * 3 * 6, 6, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)

        elif input_type == 'key_skeleton_sub' and window_size == 9 and n_layer == 2:
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=dropout))

            self.fc = torch.nn.Linear(64 * 2 * 19, 6, bias=True)

            torch.nn.init.xavier_uniform_(self.fc.weight)


        else :
            print("model input type error")


    def forward(self, x):
        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        if self.input_type == 'skeleton_sub' and self.window_size == 24 and self.n_layer == 4:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(-1, 256 * 3 * 12)

        elif self.input_type == 'skeleton_sub' and self.window_size == 15 and self.n_layer == 2:
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(-1, 64 * 3 * 7)

        elif self.input_type == 'skeleton_sub' and self.window_size == 9 and self.n_layer == 4:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(-1, 256 * 3 * 4)

        elif self.input_type == 'skeleton_sub' and self.window_size == 9 and self.n_layer == 2:
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(-1, 64 * 3 * 4)

        elif self.input_type == 'skeleton_angle' and self.window_size == 9 and self.n_layer == 2:
          #shape 확인 torch.Size([128, 1, 13, 27])
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(-1, 64 * 3 * 6)


        elif self.input_type == 'skeleton_angle' and self.window_size == 15 and self.n_layer == 2:
          #shape 확인 torch.Size([128, 1, 13, 45])
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(-1, 64 * 3 * 11)


        elif self.input_type == 'key_skeleton_sub' and self.window_size == 9 and self.n_layer == 2:
          #shape 확인 torch.Size([128, 1, 13, 45])
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(-1, 64 * 2 * 19)


        out = self.fc(out) 
        
        return out
