import math
import time
import util
import torch
import logging
import argparse

import numpy as np
from torch import nn
import torch.optim as optim

from util import DataLoaderS
from model import *
from model_time_shift import A2GCN


logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='logging_ablation.txt',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s : %(message)s',
                    )

logging.info('\n\n\n*********************************start*************************\n\n\n')
device = torch.device('cuda')

##solar 137
##traffic 862
##electricity 321
###exchange-rate 8

parser = argparse.ArgumentParser()


# --num_nodes 137 --forecast_horizon 12 --data data/single_data/solar-energy/solar_AL.txt --nhid 16 --gnn_channel 32 --attentional_relation_channel 128

parser.add_argument('--num_nodes',type=int,default=137,help='')
parser.add_argument('--forecast_horizon',type=int,default=12,help='')
parser.add_argument('--data',type=str,default='./data/single_data/solar-energy/solar_AL.txt',help='')

parser.add_argument('--sparse',type=float,default=0.1,help='')
parser.add_argument('--nhid',type=int,default=16,help='')
parser.add_argument('--gnn_channel',type=int,default=32,help='')
parser.add_argument('--attentional_relation_channel',type=int,default=128,help='')
args = parser.parse_args()



Data = DataLoaderS(args.data, 0.6, 0.2, device, horizon=12,window=24*7,normalize = 2 )
num_nodes = args.num_nodes #137
sparse = args.sparse #0.1
batch_size = 64
model = A2GCN(num_nodes, in_T = 24*7, in_dim = 1,out_T=1,out_dim=1,
                predefined_G=None, \
                 channel =  args.nhid, gnn_channel=args.gnn_channel,attentional_relation_channel=args.attentional_relation_channel,sparse = int(num_nodes*sparse),gnn_layers=2,dropout=0,device=device,)


# t_shift_net(device, num_nodes=num_nodes, T=24*7,delta_T=24*7,dropout=0.3, supports=None, \
#              in_dim=1, out_dim=1, residual_channels=16, \
#              skip_channels=256, end_channels=512,layers = 1,sparse = sparse,)
model = model.to(device)

optimizer = optim.Adam([
            {'params':filter(lambda x:id(x) not in [id(j) for j in model.latent_graph.parameters()], model.parameters()), 'lr':1e-3, 'weight_decay':0},
            {'params':model.latent_graph.parameters(),'lr':1e-2},
                                    ])

evaluateL2 = nn.MSELoss(size_average=False).to(device)
evaluateL1 = nn.L1Loss(size_average=False).to(device)                                

logging.info('\n\n\n*********************************start*************************\n\n\n')

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = X.unsqueeze(dim = 1).permute(0,1,3,2)

        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)
    

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation

for ep in range(60):
    print('*******{}*******'.format(ep))
    logging.info('*******{}*******'.format(ep))
    
    losses = []
    model.train()
    start = time.time()
    for x,y in Data.get_batches(Data.train[0], Data.train[1],batch_size,True):
        optimizer.zero_grad()
        x = x.unsqueeze(dim = 1).permute(0,1,3,2)
        out = model(x)
        out = out.squeeze()
        scale = Data.scale.unsqueeze(dim = 0)
        loss = evaluateL1(out*scale,y*scale)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        losses.append(loss.item())
        if len(losses)%50 == 0:
            print(np.mean(losses))
    now = time.time()
    print('train epoch time: {:.2f} \s'.format(now- start))
    logging.info('train epoch time: {:.2f} \s'.format(now- start))
    
    r1 = evaluate(Data,Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,batch_size)
    r2 = evaluate(Data,Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,batch_size)

    print('inference time: {:.2f} \s'.format(0.5*(time.time()- now)))

    logging.info(' '.join([str(i) for i in r1+r2]))
    print(r1,r2)