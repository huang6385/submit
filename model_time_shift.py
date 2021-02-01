import sys
import time 
import torch
import random 

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util_torch import graphs_re,choice_by_prob,gumbel_softmax,graphs_threshold


class GNN_Block(nn.Module):
    def __init__(self,input_size,output_size,gnn_layers = 1,dropout =0.3):
        super(GNN_Block,self).__init__()
        
        self.input_size  = input_size
        self.output_size = output_size
        
        self.gnn_layers = gnn_layers
        self.dropout = dropout
   
        #gcn init
        self.MLP = nn.ModuleList()
        for i in range(self.gnn_layers):
            if i == 0:
                self.MLP.append(
                    nn.Sequential(nn.Linear(input_size,output_size),
                                 nn.PReLU(),
                                 nn.Dropout(self.dropout))
                                 )
            else:
                self.MLP.append(
                    nn.Sequential(nn.Linear(output_size,output_size),
                                  nn.PReLU(),
                                  nn.Dropout(self.dropout))
                                 )
            
            nn.init.xavier_normal_(self.MLP[-1][0].weight)
        
    def forward(self,x,adj):
        '''
        x.shape: B*channel*N*T
        adj.shape: N*N
        '''
        B,N,c = x.shape
        
        out = [x]
        # layers gnn
        for i in range(self.gnn_layers):
            ##gnn block
            if len(adj.shape) == 2:
                x = torch.einsum('nm,bmk->bnk',(adj,out[-1]))
            else:
                x = torch.einsum('bnm,bmk->bnk',(adj,out[-1]))

            x = x.contiguous() #B*N*(T*rnn_dim)
            x = self.MLP[i](x)
            
            out.append(x)
            
        out = out[1:] #[B*N*k,B*N*k,.....]
        out = torch.stack(out,dim = 2) #B*N*layers*k
        
        return out
        




    
class Multi_embed(nn.Module):
    def __init__(self,multi_embeddings):
        '''
        multi_embeddings: [] e.g. [[24,4],[7,2],[12,3]->[[hour embedding],[weekday embedding],[month embedding]]
        '''
        super(Multi_embed,self).__init__()
        
        self.embeddings = nn.ModuleList()
        for vocab_size,dim in multi_embeddings:
            self.embeddings.append(nn.Embedding(vocab_size,dim))
            nn.init.xavier_normal_(self.embeddings[-1].weight)
        
        
        
    def forward(self,x):
        
        B,N,T,c = x.shape
        
        res = []
        for i in range(x.shape[-1]):
            ### the input of embedding must be LongTensor
            res.append(self.embeddings[i](x[:,:,:,i].long()))
        
        return torch.cat(res,dim = -1) #B*N*T*-1
            

    
    
class Auto_graph_learner(nn.Module):
    def __init__(self,num_nodes,sparse,agl_dim = 32):
        '''
        num_nodes: 
        sparse: sparse number (int)
        
        '''
        super(Auto_graph_learner,self).__init__()

        self.num_nodes = num_nodes
        self.sparse = sparse
        self.new_supports = nn.Parameter(torch.ones(num_nodes, num_nodes), requires_grad=True)
        self.agl_node = nn.Parameter(torch.randn(num_nodes,agl_dim))
        self.agl_dim =agl_dim
        nn.init.xavier_normal_(self.agl_node)
        
    def forward(self,):
         ### self.sparse set: about 20-30node is resonable
        new_supports = self.new_supports #torch.matmul(self.nodevec1, self.nodevec2)
#         new_supports = torch.einsum('nk,mk->nm',self.agl_node,self.agl_node)/(self.agl_dim**0.5)
        if self.training:
            new_supports = F.softmax(new_supports,dim = -1)
            new_supports = torch.log(new_supports)
            cur = torch.zeros_like(new_supports)
            for i in range(int(self.sparse)):
                cur += gumbel_softmax(new_supports,0.5)

            new_supports = cur/float(int(self.sparse))
            
        else:
            new_supports = graphs_re(new_supports,sparse =  self.sparse)
            new_supports[new_supports<=0] = -1e15
            new_supports = F.softmax(new_supports, dim=-1)
            
        return new_supports
            

        
class Attentional_relation_learner(nn.Module):
    def __init__(self,num_nodes,group_k,input_dim,hidden_dim,out_dim,qkdim=128):
        '''
        Node pay more attention to more appropriate relation/graph (e.g. Identity, Latent graph, Predifined graph)
        
        
        num_nodes:
        group_k: relation count (layers*relations)
        input_dim: input dimension
        hidden_dim:
        out_dim: output dimension
        '''
        super(Attentional_relation_learner, self).__init__()
        
        self.num_nodes = num_nodes
        self.group_k = group_k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.qkdim = qkdim
        self.Weight_q = nn.Parameter(torch.randn(self.num_nodes,self.qkdim),requires_grad=True)
        self.W_k = nn.Linear(self.input_dim,self.qkdim)
        self.W_v = nn.Sequential(
                                nn.Linear(self.group_k*self.input_dim,self.hidden_dim), 
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim,self.out_dim),
                                )
        
        nn.init.xavier_normal_(self.Weight_q)
        nn.init.xavier_normal_(self.W_k.weight)
        nn.init.xavier_normal_(self.W_v[0].weight)
        nn.init.xavier_normal_(self.W_v[2].weight)

        
        
    def forward(self,x):
        '''
        '''
        B,N,group_k,input_dim=x.shape
        if group_k != self.group_k or input_dim!=self.input_dim:assert False
        
        K,V = x,x
        Q = self.Weight_q #N*k_dim
        K = self.W_k(K) #B*N*L*k_dim
        attention = torch.einsum('nk,bnlk->bnl',Q,K)/(self.qkdim**0.5)
        attention = F.softmax(attention,dim = -1)
        
        V = attention.reshape(B,N,-1,1) * V
#         V = torch.einsum('bng,bngk->bnk',attention,V)
        V = V.reshape(B,N,-1)
        V = self.W_v(V) #B*N*out_dim

        return V
    
    
class LSTM_skip(nn.Module):
    def __init__(self,in_channel,out_channel,window = 24):
        super(LSTM_skip,self).__init__()
        self.window = window

        self.rnncell = nn.LSTMCell(in_channel,out_channel)
        self.w_skip = nn.Linear(out_channel,out_channel)

        self.h0 = nn.Parameter(torch.randn(1, out_channel))
        self.c0 = nn.Parameter(torch.randn(1, out_channel))
        
        nn.init.xavier_normal_(self.w_skip.weight)
        nn.init.xavier_normal_(self.h0)
        nn.init.xavier_normal_(self.c0)

    def forward(self,x):
        B,T,c = x.shape
        
        hx = self.h0.repeat(B,1)
        cx = self.c0.repeat(B,1)
        output = []
        for i in range(T):
            if i>=self.window:
                hx, cx = self.rnncell(x[:,i], (hx+self.w_skip(output[i-self.window]), cx))
            else:
                hx, cx = self.rnncell(x[:,i], (hx, cx))
            output.append(hx)
        
        output = torch.stack(output,axis = 1)
        return output,(hx,cx)       
            
class A2GCN(nn.Module):
    def __init__(self,num_nodes,in_T,in_dim,out_T,out_dim=1,multi_embeddings=None,predefined_G=None,
                 sparse=15,agl_dim = 32,
                 cnn_kernel = 3,
                 gnn_layers=2,dropout=0.3,channel=32,gnn_channel=None,attentional_relation_channel=None,
                 device=None): #,**kwargs):
        '''
        num_nodes:
        in_T: historical length
        in_dim:
        out_T: forecasting future length
        out_dim: forecasting future dimension, usually is 1
        multi_embeddings: when the input have some temporal discrete feature, like month, weekday, holiday, etc.
                        need to notice that, the input must --->      continuous features (k-dim) ||cat   discrete features (in_T - k-dim)
        predefined_G: user predifined graph
        sparse (int): learnable matrix/graph sparsity
        gnn_layers: the layers of gnn
        dropout:
        channel: use this part to set all the hidden dimension of A2GCN
        device: cpu or cuda:0 or cuda:1
        '''
        super(A2GCN, self).__init__()
        

        self.num_nodes = num_nodes
        self.in_T = in_T
        self.in_dim = in_dim
        self.out_T = out_T
        self.out_dim = out_dim
        channel = channel
        gnn_channel = channel*8 if gnn_channel is None else gnn_channel
        attentional_relation_channel = channel*16 if attentional_relation_channel is None else attentional_relation_channel
        
        self.multi_embeddings = multi_embeddings
        
        if self.multi_embeddings is not None:
            self.embeds = Multi_embed(multi_embeddings)
            self.start_fc = nn.Linear(
                in_dim-len(self.multi_embeddings)+sum([j for i,j in self.multi_embeddings]), channel)
            self.start_cnn = torch.nn.Conv2d(in_dim-len(self.multi_embeddings)+sum([j for i,j in self.multi_embeddings]),channel,kernel_size=(1,cnn_kernel),padding = (0,cnn_kernel//2))
        else:
            self.embeds = None
            self.start_fc = nn.Linear(in_dim,channel)
            self.start_cnn = torch.nn.Conv2d(in_dim,channel,kernel_size=(1,3),padding = (0,1))
        
            
        self.start_rnn = nn.LSTM(channel,channel,batch_first = True)
#         self.start_rnn = LSTM_skip(channel,channel)
        
        self.latent_graph = Auto_graph_learner(num_nodes,sparse,agl_dim)
        self_adj = nn.Parameter(torch.eye(num_nodes,num_nodes)*1.0, requires_grad=False)
        
        self.graphs = predefined_G if predefined_G is not None else []
        self.len_pre = len(self.graphs)
        self.graphs.append(self_adj)
        self.graphs = [i.to(device) for i in self.graphs]
        print('len graphs is {}'.format(len(self.graphs)))
        
        self.node_embedding = nn.Parameter(torch.randn(2,num_nodes,32))
        
        self.gnns = nn.ModuleList()
        for i in range(len(self.graphs)+1):
            self.gnns.append(GNN_Block(in_T*channel,gnn_channel,gnn_layers,dropout))
            
        
        #### out_dim must equal to 1.
        self.relation_learner = Attentional_relation_learner(num_nodes,gnn_layers*(len(self.graphs)+1),gnn_channel,attentional_relation_channel,self.out_T*self.out_dim)
        
        
        nn.init.xavier_normal_(self.start_fc.weight)
        nn.init.xavier_normal_(self.start_cnn.weight)
        nn.init.xavier_normal_(self.node_embedding)
        
        
    def forward(self, x):
        if x.shape[3]<self.in_T:
            x = nn.functional.pad(x,(self.in_T-x.shape[3],0,0,0))
        B,channel,N,T = x.shape
        x = x.permute(0,2,3,1).contiguous()
        
        if self.multi_embeddings is not None:
            temp = len(self.multi_embeddings)
            #discrete features embedding
            x = torch.cat([x[:,:,:,:-temp],self.embeds(x[:,:,:,-temp:])],dim = -1)
           
        #x = self.start_fc(x)
        
        # use cnn for data
        x = x.permute(0,3,1,2) #->B*C*N*T
        x = self.start_cnn(x)
        x = x.permute(0,2,3,1)
        
        x = x.reshape(B*N,T,-1)
        x,(hn,cn) = self.start_rnn(x)
        # x = self.start_T_att(x)
        x = x.contiguous().reshape(B,N,-1)
        
        relation = torch.einsum('ink,imk->inm',self.node_embedding,self.node_embedding)/(32**0.5)
        
        groups_fea=[]
        for i in range(len(self.gnns)):
            _adj = self.graphs[i] if i != len(self.gnns)-1 else self.latent_graph()
            if i <self.len_pre:
                _adj = (_adj>0)*relation[i]
            groups_fea.append(self.gnns[i](x,_adj))
                
        groups_fea = torch.cat(groups_fea,dim = 2)
        out = self.relation_learner(groups_fea)

        out = out.reshape(B,N,self.out_T,self.out_dim).permute(0,2,1,3).contiguous()
        return out

if __name__ == '__main__':
    test_model = A2GCN(num_nodes=100,in_T=12,in_dim=3,out_T=6,out_dim=1,multi_embeddings=None,predefined_G=[torch.randn(100,100)],sparse=15,gnn_layers=2,dropout=0.3,channel=32)
    x = torch.randn(5,3,100,12)
    out = test_model(x)
    print(out.shape)