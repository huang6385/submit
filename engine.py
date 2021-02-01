import torch.optim as optim
from model import *
from model_time_shift import A2GCN
from stgnn import STGNN
from StemGNN import Model
import util

model_dict = {'graph_wavenet':gwnet,
            'a2gcn':A2GCN}

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, device, supports,layers,sparse,**kwargs):
        # self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
#         self.model = M(device, num_nodes, dropout=dropout, supports=supports, \
#             in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, \
#              skip_channels=nhid * 8, end_channels=nhid * 16,layers=layers,sparse = sparse,**kwargs)
        self.loss = util.masked_mae    
        if kwargs['model'] == 'a2gcn':
            self.model = A2GCN(num_nodes, in_T = seq_length, in_dim = in_dim,out_T=12,out_dim=1,
                           multi_embeddings=[[289,30],[7,2]],predefined_G=supports, \
                     channel =nhid, sparse = int(num_nodes*sparse),gnn_layers=layers,dropout=dropout,device=device,)
            self.model.to(device)
            
            base_params = list(map(id,self.model.latent_graph.parameters()))
            other = filter(lambda x:id(x) not in base_params, self.model.parameters())
            self.optimizer = optim.Adam([
                        {'params':other, 'lr':kwargs['learning_rate'], 'weight_decay':kwargs['weight_decay']},
                        {'params':self.model.latent_graph.parameters(),'lr':kwargs['latent_graph_lr']},
                                                ])
        elif kwargs['model'] == 'stgnn':
            self.model = STGNN(num_nodes, in_T = seq_length, in_dim = in_dim,out_T=12,out_dim=1,
                           multi_embeddings=[[289,30],[7,2]],predefined_G=supports, \
                     channel =nhid, sparse = int(num_nodes*sparse),gnn_layers=layers,dropout=dropout,device=device,)
            self.model.to(device)
            
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-3)
        elif kwargs['model'] == 'stemgnn':
            self.model = Model(units=num_nodes, stack_cnt=2, time_step=seq_length, multi_layer=5, horizon=12, dropout_rate=0.3, leaky_rate=0.2,device=device)
            self.model.to(device)
            
            self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=1e-4)
            self.loss = util.masked_rmse
            
        
        self.scaler = scaler
        self.clip = 5
        self.kwargs = kwargs

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        

        loss = self.loss(predict, real, 0.0) #+ torch.mean(torch.abs(self.model.cache))

        # if self.kwargs['model'] == 't_shift_net':
        #     attention = self.model.get_cache()
        #     loss -= 1*torch.sum(attention**2)
        

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            output = output.transpose(1,3)
            #output = [batch_size,12,num_nodes,1]
            real = torch.unsqueeze(real_val,dim=1)
            predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse


    def kl_loss(self,input,target):
        return torch.sum(target*torch.log(target+1e-10)-target*torch.log(input+1e-10))