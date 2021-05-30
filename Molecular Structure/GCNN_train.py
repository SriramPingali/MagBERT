from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

import json
import numpy as np
from collections import defaultdict

visited = {}
type_ = []
for line in open('2zoq.pdb'):
    list = line.split()
    id = list[0]
    if id == 'CONECT':
        type_.append(list[1:])
    #print(type_)

max_1 = int(0)
for i in type_:
    for j in i:
        if(int(max_1)<int(j)):
            max_1 = j

matrix = np.zeros((int(max_1)+1,int(max_1)+1))

adj_list_old=[]
for i in type_:
    for j in i:
        for k in i:
            if(j!=k):
                adj_list_old.append([j,k])

adj_list =[]
for i in adj_list_old:
    if i not in adj_list: 
        adj_list.append(i)

adj_list = sorted(adj_list)

store={}
for i in adj_list:
    #print(i[0],i[1])
    
    if i[0] not in store:
        store[i[0]]=[int(i[1])]
    else:
        l=store[i[0]]
        if i[1] not in l:
            l.append(int(i[1]))
            store[i[0]]=l

ans=[]
for key,value in store.items():
    temp=[]
    temp.append(int(key))
    temp+=value
    ans.append(temp)

for i in ans:
    j=92-len(i)
    for k in range(0,j):
        i.append(0)

dict_label = {}
list_1 = []
list_11 = []
for line in open('2zoq.pdb'):
    list = line.split()
    id = list[0]
    if id == 'ATOM' or id == 'HETATM':
        #print(list[1])
        list_1.append(list[1])
        list_11.append(list[11])

dict_label = {k:v for k,v in zip(list_1,list_11)}

label_temp = []
for i in ans:
    #print(i[0])
    label_temp.append(dict_label[str(i[0])])

label = []
for i in label_temp:
    if i == 'C':
        label.append(0)
    elif i == 'I':
        label.append(1)
    elif i == 'N':
        label.append(2)
    elif i == 'NA':
        label.append(3)
    elif i == 'O':
        label.append(4)
    elif i == 'P':
        label.append(5)
    elif i == 'S':
        label.append(6)

import pandas as pd
feature_temp_1 = pd.read_csv('feature_temp.csv')
feature_temp = feature_temp_1.fillna('NA')

list_feature_map_temp = []
list_feature_map_value_temp = []
for line in open('2zoq.pdb'):
    list = line.split()
    id = list[0]
    if id == 'ATOM' or id == 'HETATM':
        #print(list[1])
        list_feature_map_temp.append(list[1])
        list_feature_map_value_temp.append(list[3])
dict_label_feature = {k:v for k,v in zip(list_feature_map_temp,list_feature_map_value_temp)}
temp = []
for i in ans:
    #print(i[0])
    temp.append(dict_label_feature[str(i[0])])
temp_temp = []
for i in temp:
    temp_temp.append(feature_temp[feature_temp['Symbol'] == i].values.tolist())
temp_temp_new = []
for i in range(0,len(temp_temp)):
    temp_temp_new.append(temp_temp[i][0][2:8])




# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train_1, idx_val, idx_test = load_data("/Users/ashwani/Documents/BIO_NLP_GPU/pygcn-master/data/cora/")
adj = torch.tensor(ans,dtype=torch.float32)
labels = torch.tensor(label)
features = torch.tensor(temp_temp_new)


print("adj=",adj.shape)
print("labels=",labels.shape)
print("features=",features.shape)
idx_train = idx_train_1[0:30]
idx_val = idx_train_1[30:60]
idx_test = idx_train_1[60:92]
print("idx_train=",idx_train)
print("idx_test=",idx_test)
print("idx_val=",idx_val)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=max(label) + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    output_1=torch.tensor()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    print("output",output)
    output_1 = output
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#print(output_1)

# Testing
test()


import json
import numpy as np
from collections import defaultdict

visited = {}
type_ = []
for line in open('4qtc.pdb'):
    list = line.split()
    id = list[0]
    if id == 'CONECT':
        type_.append(list[1:])
    #print(type_)

max_1 = int(0)
for i in type_:
    for j in i:
        if(int(max_1)<int(j)):
            max_1 = j

matrix = np.zeros((int(max_1)+1,int(max_1)+1))

adj_list_old=[]
for i in type_:
    for j in i:
        for k in i:
            if(j!=k):
                adj_list_old.append([j,k])

adj_list =[]
for i in adj_list_old:
    if i not in adj_list: 
        adj_list.append(i)

adj_list = sorted(adj_list)

store={}
for i in adj_list:
    #print(i[0],i[1])
    
    if i[0] not in store:
        store[i[0]]=[int(i[1])]
    else:
        l=store[i[0]]
        if i[1] not in l:
            l.append(int(i[1]))
            store[i[0]]=l

ans=[]
for key,value in store.items():
    temp=[]
    temp.append(int(key))
    temp+=value
    ans.append(temp)

for i in ans:
    j=92-len(i)
    for k in range(0,j):
        i.append(0)

dict_label = {}
list_1 = []
list_11 = []
for line in open('4qtc.pdb'):
    list = line.split()
    id = list[0]
    if id == 'ATOM' or id == 'HETATM':
        #print(list[1])
        list_1.append(list[1])
        list_11.append(list[11])

dict_label = {k:v for k,v in zip(list_1,list_11)}

label_temp = []
for i in ans:
    #print(i[0])
    label_temp.append(dict_label[str(i[0])])

label = []
for i in label_temp:
    if i == 'C':
        label.append(0)
    elif i == 'I':
        label.append(1)
    elif i == 'N':
        label.append(2)
    elif i == 'NA':
        label.append(3)
    elif i == 'O':
        label.append(4)
    elif i == 'P':
        label.append(5)
    elif i == 'S':
        label.append(6)

import pandas as pd
feature_temp_1 = pd.read_csv('feature_temp.csv')
feature_temp = feature_temp_1.fillna('NA')

list_feature_map_temp = []
list_feature_map_value_temp = []
for line in open('4qtc.pdb'):
    list = line.split()
    id = list[0]
    if id == 'ATOM' or id == 'HETATM':
        #print(list[1])
        list_feature_map_temp.append(list[1])
        list_feature_map_value_temp.append(list[3])
dict_label_feature = {k:v for k,v in zip(list_feature_map_temp,list_feature_map_value_temp)}
temp = []
for i in ans:
    #print(i[0])
    temp.append(dict_label_feature[str(i[0])])
temp_temp = []
for i in temp:
    temp_temp.append(feature_temp[feature_temp['Symbol'] == i].values.tolist())
#temp_temp_1 = [x for x in temp_temp if x]
temp_temp_new = []
for i in range(0,len(temp_temp)):
    temp_temp_new.append(temp_temp[i][0][2:8])


# Load data
adj, features, labels, idx_train_1, idx_val, idx_test = load_data("/Users/ashwani/Documents/BIO_NLP_GPU/pygcn-master/data/cora/")
adj = torch.tensor(ans,dtype=torch.float32)
labels = torch.tensor(label)
features = torch.tensor(temp_temp_new)


print("adj=",adj.shape)
print("labels=",labels.shape)
print("features=",features.shape)
idx_train = idx_train_1[0:30]
idx_val = idx_train_1[30:60]
idx_test = idx_train_1[60:92]
print("idx_train=",idx_train)
print("idx_test=",idx_test)
print("idx_val=",idx_val)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=max(label) + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    output_2=torch.tensor()

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#print(output_1)

# Testing
test()

output_3 = output_1 + output_2
print(output_3)

