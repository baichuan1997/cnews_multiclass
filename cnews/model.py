# TextRNN Model
import torch
from torch import nn
import torch.nn.functional as F
 
# 文本分类，RNN模型
class TextRNN(nn.Module):   
    def __init__(self):
        super(TextRNN, self).__init__()
        # 进行词嵌入
        self.embedding = nn.Embedding(5000, 64)  
        # self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.f1 = nn.Sequential(nn.Linear(256,128),
                                nn.Dropout(0.2),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128,10))
 
    def forward(self, x):
        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = F.dropout(x,p=0.2)
        x = self.f1(x[:,-1,:])
        return self.f2(x)
 
