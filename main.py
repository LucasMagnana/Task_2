import json
import os
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

from NN import NN

data = {'X':[], 'y':[]}
for root, dirs, files in os.walk("./data", topdown=False):
   for name in files:
        if("dataset" in name):
            with open(os.path.join(root, name), 'r') as infile:
                d = json.load(infile)
                data['X'] += d['X']
                data['y'] += d['y']


network = NN(len(data['X'][0])-1, len(data['y'][0]))

optimizer = torch.optim.Adam(network.parameters(), lr=1e-6)
loss = nn.CrossEntropyLoss()

data_train = {'X':[], 'y':[]}
data_train['X'] = data['X'][int(len(data['X'])*20/100):]
data_train['y'] = data['y'][int(len(data['y'])*20/100):]

data_test = {'X':[], 'y':[]}
data_test['X'] = data['X'][:int(len(data['X'])*20/100)]
data_test['y'] = data['y'][:int(len(data['y'])*20/100)]

print(len(data_train['X']), len(data_test['X']))

tab_loss = []

for i in range(150000):
    X = []
    y = []
    for _ in range(32):
        r = random.randint(0, len(data_train['X'])-1)
        X.append(data_train['X'][r][:-1])
        y.append(data_train['y'][r].index(1.0))

    tens_X = torch.Tensor(X)
    tens_y = torch.Tensor(y).long()

    optimizer.zero_grad()
    output = network(tens_X)
    l = loss(output, tens_y)
    tab_loss.append(l.item())
    l.backward()
    optimizer.step()


nb_test = 0
nb_good_prediction = 0
for i in range(len(data_test['X'])):
    X = data_test['X'][i][:-1]
    y = data_test['y'][i].index(1.0)

    tens_X = torch.Tensor(X)
    tens_y = torch.Tensor(y).long()

    output = network(tens_X)
    pred = output.argmax(dim=0, keepdim=True)
    if(pred.item() == y):
        nb_good_prediction += 1
    nb_test += 1

print(nb_good_prediction/nb_test*100, "%")



plt.plot(tab_loss)
plt.ylabel('Error')
plt.show()

