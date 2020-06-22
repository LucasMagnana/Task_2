import json
import os
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import sys

from NN import NN


def test(data_test, network): #function used to test a network on a dataset
    nb_test = 0
    nb_good_prediction = 0
    for i in range(len(data_test['X'])): #for each data in the dataset
        X = data_test['X'][i][:-1] #delete the total duration
        y = data_test['y'][i].index(1.0) #find the number of the class thanks to the one hot encoding

        tens_X = torch.Tensor(X) 
        tens_y = torch.Tensor(y).long()

        output = network(tens_X) #pass the input in the network
        pred = output.argmax(dim=0, keepdim=True)#find the higher probability in the output
        if(pred.item() == y): #if the prediction is correct
            nb_good_prediction += 1 #one more good prediction
        nb_test += 1 #one more data tested
    return nb_good_prediction/nb_test #we return the ratio between the good predictions and the total predictions


data = {'X':[], 'y':[]}
for root, dirs, files in os.walk("./data", topdown=False): #load all the data
   for name in files:
        if("dataset" in name):
            with open(os.path.join(root, name), 'r') as infile:
                d = json.load(infile)
                data['X'] += d['X']
                data['y'] += d['y']

if(len(sys.argv) > 1 and sys.argv[1] == 'test'): #if there is the argument "test"
    print("Testing network.pt...")

    network = NN(len(data['X'][0])-1, len(data['y'][0])) #load the network in network.pt
    network.load_state_dict(torch.load("./network.pt"))
    network.eval() 

    g_predict = test(data, network) #test the network on all the dataset
    print(g_predict*100, "%") #print the result

else : #else if there is not the argument test
    print("Training a new network...")
    network = NN(len(data['X'][0])-1, len(data['y'][0])) #create a new network

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-6) #define the optimizer
    loss = nn.CrossEntropyLoss() #define the loss function

    data_test = {'X':[], 'y':[]}

    for i in range(int(len(data['X'])*20/100)): #take randomly 20% of the dataset for the cross-validation
        r = random.randint(0, len(data['X'])-1)
        data_test['X'].append(data['X'][r])
        data_test['y'].append(data['y'][r])
        data['X'].pop(r)
        data['y'].pop(r)

    data_train = data #the rest of the data are for the training

    print(len(data_train['X']), len(data_test['X'])) #print the size of the two dataset

    tab_loss = []

    for i in range(175000):
        X = []
        y = []
        for _ in range(32): #create randomly a batch of 32 inputs
            r = random.randint(0, len(data_train['X'])-1)
            X.append(data_train['X'][r][:-1])
            y.append(data_train['y'][r].index(1.0))

        tens_X = torch.Tensor(X)
        tens_y = torch.Tensor(y).long()

        optimizer.zero_grad()
        output = network(tens_X) #pass the batch in the network
        l = loss(output, tens_y) #calculate the error
        tab_loss.append(l.item()) #save the error to plot a graph
        l.backward() 
        optimizer.step() #make the network learn


    g_predict = test(data_test, network) #cross-validation
    print(g_predict*100, "%")

    if(g_predict > 0.89): #if the network has better results than the one already saved
        print("Saving network...")
        torch.save(network.state_dict(), "./network.pt") #save the network

    g_predict = test(data_train, network) #test the network on the training dataset
    print(g_predict*100, "%")



    plt.plot(tab_loss) #plot the graph of errors
    plt.ylabel('Error')
    plt.show()

