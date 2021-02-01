import torch
import numpy as np
import argparse
import time
import util
import logging
import matplotlib.pyplot as plt
from engine import trainer

logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='logging.txt',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s : %(message)s',
                    )

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
# parser.add_argument('--pos_emb',type=str,default = 'data/METR-LA/spatiol_embedding.npy')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--latent_graph_lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--epochs',type=int,default=100,help='') ##default setting is 100
parser.add_argument('--print_every',type=int,default=100,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--save_auto_graph',type=int,default=0,help='save/notsave auto graph learner per epoch')

parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--model',type=str,default='t_shift_net',help='model select') #graph_wavenet,t_shift_net
parser.add_argument('--sparse',type=float,default=0.1,help='model learned adj sparse rate')
parser.add_argument('--layers',type=int,default=2,help='model layers')
parser.add_argument('--model_path',type=str,default='None',help='model path, Please set in test mode!') #graph_wavenet,t_shift_net


args = parser.parse_args()
# args.adjtype = 'doubletransition'


# args.epochs = 0
# args.data ='data/PEMS-BAY'
# args.adjdata ='data/sensor_graph/adj_mx_bay.pkl'
# args.num_nodes= 325



#python3 train.py --model t_shift_net --epochs 100 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --num_nodes 325 --sparse 0.05 --layers 3
#python3 train.py --model t_shift_net --epochs 100

def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    if adj_mx is None:
        supports = None 
    else:
        supports = [torch.tensor(i).to(device) for i in adj_mx]
    
    print(args)

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                    device, supports,layers = args.layers, 
                     sparse = args.sparse,model = args.model,
                    latent_graph_lr=args.latent_graph_lr,learning_rate = args.learning_rate,weight_decay =  args.weight_decay) #,pos_emb = args.pos_emb

    # engine.model.load_state_dict(torch.load('garage/metr_epoch_66_2.73.pth'),strict=False) #

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        if args.save_auto_graph:
            torch.save(engine.model.latent_graph.new_supports,'analysis/epoch_{}_new_supports.t'.format(i))
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
                logging.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
        
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        logging.info(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        
#         his_loss.append(mvalid_loss)
#         torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        his_loss.append(mvalid_loss)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

        
        
        
        ################# test score
        engine.model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1,3)[:,0,:,:]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]
        yhat = scaler.inverse_transform(yhat)


        amae = []
        amape = []
        armse = []
        for i in range(12):
            pred = yhat[:,:,i]
            real = realy[:,:,i]
            metrics = util.metric(pred,real)
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        logging.info(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    
        ################# test score

        
        
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing 
    if args.epochs!=0:
        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
        
        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    else:
        engine.model.load_state_dict(torch.load(args.model_path))
    
    
    engine.model.eval()
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]
    yhat = scaler.inverse_transform(yhat)
    

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = yhat[:,:,i]
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        logging.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    logging.info(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    
    if args.epochs!=0:
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
