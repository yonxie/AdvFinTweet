# -*- coding: utf-8 -*-
"""
@file: attack_main
@author: Yong Xie
@time: 5/13/2021 4:28 PM
@Description: 
"""

from torch.utils.data import DataLoader
import numpy as np
import torch
import json
import warnings
import sys
import os
from collections import defaultdict
from tqdm import tqdm

sys.path.append('.')

from model.utils import weight_init, GloveEmbedding
from model.config_loader import config, path
from model.data_loader import StockData
from model.stock_net import StockNet, StocknetWrapper
from model.tweet_gru import TweetGRU, TweetGRUWrapper
from model.tweet_lstm import TweetLSTM, TweetLSTMWrapper
from model.han import HierAttNet, HANWrapper


warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.double)

np.random.seed(42)
device = config['model']['device']
emb_dev = config['model']['device']

experiment_name = '{}_{}_{}_gsize{}_hsize{}_alpha{}_nmsg{}_nword{}_lr{}_wd{}_schedule{}_{}'.format(
    config['model']['prefix'], config['model']['name'], config['stage'], config['model']['g_size'], config['model']['h_size'], config['model']['alpha'],
    config['model']['max_n_msgs'], config['model']['max_n_words'], config['train']['lr'],
    config['train']['weight_decay'], config['train']['schedule_step'], config['train']['schedule_gamma'] 
)

print('-- RUNNING EXPERIENMENT: {} --'.format(experiment_name))

if config['model']['word_embed_type'] == 'glove':
    embedding = GloveEmbedding(resource=path.glove, n_d=config['model']['word_embed_size'], 
                   vocab_list=config['vocab'])
    embedding.to(device)

if config['model']['name'] == 'stocknet':
    network = StockNet(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                    msg_hdn_size=50, embedding=embedding, device=device, config=config).to(device).double()
    network.apply(weight_init)
    model = StocknetWrapper(experiment_name, network, 'Adam', 1, config)
    
elif config['model']['name'] == 'tweetgru':
    network = TweetGRU(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                    msg_hdn_size=50, embedding=embedding, device=device, config=config).to(device).double()
    network.apply(weight_init)
    model = TweetGRUWrapper(experiment_name, network, 'Adam', 1, config)
    
elif config['model']['name'] == 'tweetlstm':
    network = TweetLSTM(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                    msg_hdn_size=50, embedding=embedding, device=device, config=config).to(device).double()
    network.apply(weight_init)
    model = TweetLSTMWrapper(experiment_name, network, 'Adam', 1, config)

elif config['model']['name'] == 'han':
    network = HierAttNet(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                    msg_hdn_size=50, embedding=embedding, device=device, config=config).to(device).double()
    network.apply(weight_init)
    model = HANWrapper(experiment_name, network, 'Adam', 1, config)
else:
    raise Exception('Found unknow model name!!!!!!!')
model.setup(True)
    
tickers = []
for indu in config['stocks'].values():
    tickers += indu
param_norm = defaultdict(list)

# log and trajectory
if not os.path.exists('./log/train/{}'.format(experiment_name)):
    os.mkdir('./log/train/{}'.format(experiment_name))
if not config['train']['resume']:   # start training from scratch
    start_epoch = 0
else:
    start_epoch = model.load_model(ckpt_path='./checkpoints/', epoch=None, load_optimizer=True)

print('finished initializing model...')    

if config['stage'] == 'train':
    
    with open('./log/train/{}/config.txt'.format(experiment_name), 'w') as f:
        f.write(json.dumps({'model':config['model'], 'dates':config['dates'], 
                            'stocks': config['stocks'], 'train':config['train']}))
    log = open('./log/train/{}/log.txt'.format(experiment_name), 'a')
    log.flush()

print('{} on :'.format(config['stage']))
print(tickers)

if config['stage'] == 'train':
    
    # load train dataset
    train_dset = StockData(config['dates']['train'], tickers, device)
    train_loader = DataLoader(train_dset, 32, shuffle=True)
    print('training datset for dates {} containing {} instance'.format(config['dates']['train'], len(train_dset)))
    
    inputs= next(iter(train_loader))
    
# load test dataset
test_dset = StockData(config['dates']['test'], tickers, device)
test_loader = DataLoader(test_dset, 64, shuffle=False)
print('test datset for dates {} containing {} instance'.format(config['dates']['test'], len(test_dset)))

# training phase
if config['stage'] == 'train':

    for epoch in tqdm(range(start_epoch, config['train']['epochs'])):
        
        train_acc, train_f1, train_pos, train_mles = model.run(train_loader, 1, interval=50)
        test_acc, test_f1, test_pos, test_mles = model.run(test_loader, 0, interval=100)
        report = '| epoch:{} | train_acc:{:.4f}, train_pos_rate:{:.3f}, train mle: {:.4f} | test_acc:{:.3f}, test_pos:{:.3f}, test mle: {:.4F}'.format(epoch, 
                    train_acc, train_pos, np.mean(train_mles), test_acc, test_pos, np.mean(test_mles))
        print(report)
        log.write(report + '\n')
        log.flush()
      
        if epoch % 1 == 0 and epoch >= 0:
            model.save_model('./checkpoints'.format(experiment_name), epoch, save_optimizer=True)
            
    # save the results
    log.close()
    
if config['stage'] == 'test':
    epoch = 9
    model.load_model(ckpt_path='./checkpoints/', epoch=epoch, load_optimizer=True)
    acc, pos = model.run(test_loader, 0, interval=20)
    print('checkpoint: {}, acc: {}, pos rate: {}'.format(epoch, acc, pos))