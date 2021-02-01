# submit

## Data Preparation




### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).
### Step1.1: Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.


### Step2: Process raw data 

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
```
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
```

### Train Commands

```
python3 train.py --model a2gcn --nhid 16 --learning_rate 0.001 --weight_decay 0 --latent_graph_lr 0.01 --epochs 100
```



### Train Commands

```
python3 train_single.py --num_nodes 137 --forecast_horizon 12 --data data/single_data/solar-energy/solar_AL.txt --nhid 16 --gnn_channel 32 --attentional_relation_channel 128
```
