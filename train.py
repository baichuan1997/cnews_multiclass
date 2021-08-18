# 使用GRU/LSTM进行文本分类
import torch
from torch import nn
from torch import optim
import numpy as np
from model import TextRNN
from cnews_loader import read_vocab, read_category, process_file
import os
import torch.utils.data as Data #将数据分批次需要用到它


# 设置数据集目录
#train_file = 'cnews.train.small.txt'
train_file = 'cnews.train.txt'
test_file = 'cnews.test.txt'
val_file = 'cnews.val.txt'
vocab_file = 'cnews.vocab.txt'

def train():
    # 使用LSTM或者CNN
    model = TextRNN().cuda()
    # 选择损失函数
    Loss = nn.CrossEntropyLoss()
    #Loss = nn.CrossEntropyLoss()
    #Loss = nn.MSELoss() # 多分类一般不用MSE
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #optimizer = optim.Adam(model.parameters(),lr=5e-4)
    best_val_acc = 0

    # 加载之前训练过的网络参数
    if os.path.exists('model_params.pkl'):
        model.load_state_dict(torch.load('model_params.pkl'))

    for epoch in range(1000):
        # print('epoch=', epoch)
        # 将训练集分批batch
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch.cuda()
            y = y_batch.cuda()
            #print('x=', x)
            out = model(x)
            #print('out=', out)
            #print('y', y)
            loss = Loss(out,y)
            # print(f'>>> loss={loss.item():.3f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
            # print(accuracy)
        #对模型进行验证
        if (epoch+1)%1 == 0:
            # print('epoch=', epoch+1)
            i = 0
            total_acc = 0
            for step, (x_batch, y_batch) in enumerate(val_loader):
                x = x_batch.cuda()
                y = y_batch.cuda()
                # 前向传播
                out = model(x)
                accuracy = np.mean((torch.argmax(out, 1) == y).cpu().numpy())
                total_acc += accuracy
                i += 1
                # if accuracy > best_val_acc:
                    # torch.save(model.state_dict(), 'model_params.pkl')
                    # best_val_acc = accuracy
                    # print('model_params.pkl saved')
            print('epoch:{}, acc:{}'.format(epoch+1, total_acc/i))
 
# 获取文本的类别及其对应id的字典
categories, cat_to_id = read_category()
#print(categories)
#print(cat_to_id)
# 获取训练文本中所有出现过的字及其所对应的id
words, word_to_id = read_vocab(vocab_file)
#print(words)
#print(word_to_id)
#获取字数
vocab_size = len(words)
#print(vocab_size)

# 数据加载及分批
# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file(train_file, word_to_id, cat_to_id, 600)
print('x_train=', x_train)
x_val, y_val = process_file(val_file, word_to_id, cat_to_id, 600)

# 设置使用GPU
cuda = torch.device('cuda')
x_train, y_train = torch.LongTensor(x_train), torch.LongTensor(y_train)
x_val, y_val = torch.LongTensor(x_val), torch.LongTensor(y_val)
print(x_train)

train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(
    dataset=train_dataset,# torch TensorDataset format
    batch_size=64,      # 最新批数据
    shuffle=True,         # 是否随机打乱数据
    num_workers=2,        # 用于加载数据的子进程
)

val_dataset = Data.TensorDataset(x_val, y_val)
val_loader = Data.DataLoader(
    dataset=val_dataset,  # torch TensorDataset format
    batch_size=64,      # 最新批数据
    shuffle=False,         # 是否随机打乱数据
    num_workers=2,        # 用于加载数据的子进程
)


train()