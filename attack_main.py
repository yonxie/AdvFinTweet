
import sys, os
import pandas as pd
import numpy as np
sys.path.append(".") 
import torch
from torch.utils.data import DataLoader

from model.utils import GloveEmbedding, Synonymer, loss_plot, AttackMask, WordWeight
from model.config_loader import config, path
from model.data_loader import StockData
# from textfooler.textfooler import *
from model.stock_net import StockNet, StocknetWrapper
from model.tweet_gru import TweetGRU, TweetGRUWrapper
from model.tweet_lstm import TweetLSTM, TweetLSTMWrapper
from model.han import HierAttNet, HANWrapper
from sklearn.metrics import f1_score, accuracy_score
import spacy
from tqdm import tqdm


torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.double)

# set random seed
seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = config['model']['device']
emb_dev = config['model']['device']

opt = config['pgd']
experienment_name = '{}_{}_concat{}_{}_opt_{}_epoch{}_fix{}_type_{}_spa{}_alt{}_sm{}_m{}_z{}_r{}_step{}_lr{}_schedule{}_{}_proj{}'.format(
     opt['prefix'], opt['attack_corpus'], opt['concat'], opt['opt_method'], opt['model_name'], opt['ckpt_epoch'], opt['fix'], opt['attack_type'], opt['sparsity'], opt['alt'], opt['smooth'], opt['m_num'], 
     opt['z_num'], opt['r_num'], opt['steps'], opt['lr'], opt['schedule_step'], opt['schedule_gamma'], opt['projection'])

if not os.path.exists('./log/attack/{}'.format(experienment_name)):
    os.mkdir('./log/attack/{}'.format(experienment_name))

print('--EXPERIENMENT: PGD ATTACK: {}'.format(experienment_name))
print('CHECKPOINTS: ', opt['train_name'])
print('EPOCH: ', opt['ckpt_epoch'])

# load model
if config['model']['word_embed_type'] == 'glove':
    
    # embedding model
    embedding = GloveEmbedding(resource=path.glove, n_d=config['model']['word_embed_size'], 
                        vocab_list=config['vocab'])
    embedding.to(device)
    
    # synonym model
    syner = Synonymer(embedding, syn_num=15, syn_file=path.syn).to(device)
    
    # avoid mask
    tag ={'<number>', 'UNK', 'PAD', 'pad', '<ticker>', '<date>', 'must'
          '.', ',', '?', '!', '>', '<', '+', '-', '=', 
          ';', '%', '-', '&', '_', ':', '*', '...', '✓', 'inc', 'via',
          'corp'}
    
    en = spacy.load('en_core_web_sm')
    stop_words = en.Defaults.stop_words
    avoid_set = tag.union(stop_words)
    avoid_mask_layer = AttackMask(resource=path.glove, vocab_list=config['vocab'], avoid_set=avoid_set).to(device)
    
    # fin_word_mask
    fin_mask = WordWeight(resource=path.glove, vocab_list=config['vocab'], stopwords=avoid_set, config=opt)
    
    # prediction model
    if opt['model_name'] == 'stocknet':
        network = StockNet(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                        msg_hdn_size=50, embedding=embedding, device=device, config=config, syn_layer=syner).to(device).double()        
        model = StocknetWrapper(opt['train_name'], network, 'Adam', 1, config)
        
    elif opt['model_name'] == 'tweetgru':
        network = TweetGRU(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                        msg_hdn_size=50, embedding=embedding, device=device, config=config, syn_layer=syner).to(device).double()
        model = TweetGRUWrapper(opt['train_name'], network, 'Adam', 1, config)
        
    elif opt['model_name'] == 'tweetlstm':
        network = TweetLSTM(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                        msg_hdn_size=50, embedding=embedding, device=device, config=config, syn_layer=syner).to(device).double()
        model = TweetLSTMWrapper(opt['train_name'], network, 'Adam', 1, config)

    elif opt['model_name'] == 'han':
        network = HierAttNet(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                        msg_hdn_size=50, embedding=embedding, device=device, config=config, syn_layer=syner).to(device).double()
        model = HANWrapper(opt['train_name'], network, 'Adam', 1, config)
    else:
        raise Exception('Found unknow model name!!!!!!!')
    model.setup(True)
        
    model.load_model(ckpt_path='./checkpoints/', epoch=opt['ckpt_epoch'], load_optimizer=False)
    
print('Finished initializing model and embedding...')

tickers = []
for indu in config['stocks'].values():
    tickers += indu

dset = StockData(config['dates']['test'], tickers, device)
loader = DataLoader(dset, opt['batch'], shuffle=False)
print('Attack datset {} containing {} instance'.format(config['dates']['test'], len(dset)))

pgd_attack_dict = config['pgd']
pgd_attack_dict['attackable_mask_layer'] = avoid_mask_layer
pgd_attack_dict['fin_mask_layer'] = fin_mask

file = open('./log/attack/{}/stat_m1_z1.txt'.format(experienment_name), 'w')

# Attacking
tot, atk, succ_atk = 0, 0.0001, 0
print_m, print_z = 3, 3
attack_res = {}
for m in range(0, opt['m_num']+1):
    for z in range(0, opt['z_num']+1):
        attack_res[(m, z)] = {'atk': 0.00001, 'succ_atk': 0, 'attack_logs':{}}

print('Start attacking...')
for idx, sample in tqdm(enumerate(loader, 0)):
    
    return_label = sample[-1]
    
    sam, attack_res_ = model(sample, pgd_attack_dict, mode='pgd_{}_attack'.format(opt['opt_method']))

    tot += sam
    
    for m in range(0, opt['m_num']+1):
        for z in range(0, opt['z_num']+1):
            for k in attack_res_[(m,z)]['attack_log']:
                attack_res[(m,z)]['attack_logs'][tot+k] = attack_res_[(m,z)]['attack_log'][k]
                attack_res[(m,z)]['attack_logs'][tot+k]['return_label'] = return_label[k][0][-1].item()  # add return to the label
            attack_res[(m, z)]['atk'] += attack_res_[(m,z)]['atk_num']
            attack_res[(m, z)]['succ_atk'] += attack_res_[(m,z)]['succ_num']
            attack_res[(m, z)]['total'] = tot
    
    print('total: {} | acc: {:.3f} | attack: {:.0f} | pgd succ: {} | pgd rate: {:.3f}'.format(tot, attack_res[(print_m, print_z)]['atk']/tot, attack_res[(print_m, print_z)]['atk'], attack_res[(print_m, print_z)]['succ_atk'], attack_res[(print_m, print_z)]['succ_atk']/attack_res[(print_m, print_z)]['atk']))
    file.write('total: {} | acc: {:.3f} | attack: {:.0f} | pgd succ: {} | pgd rate: {:.3f}'.format(tot, attack_res[(print_m, print_z)]['atk']/tot, attack_res[(print_m, print_z)]['atk'], attack_res[(print_m, print_z)]['succ_atk'], attack_res[(print_m, print_z)]['succ_atk']/attack_res[(print_m, print_z)]['atk']))
    file.flush()

    if opt['steps'] > 0 and opt['opt_method'] == 'joint':
        loss_plot('./log/attack', experienment_name, opt['steps'])

    # collect attack results
    global_attack_results = {}
    for item in ['ASR', 'ACC', 'F1']:
        global_attack_results[item] = pd.DataFrame(np.zeros((opt['m_num']+1, opt['z_num']+1)))
    for m in range(0, opt['m_num']+1):
        for z in range(0, opt['z_num']+1):
            df_log = pd.DataFrame(attack_res[(m,z)]['attack_logs']).T
            df_log.to_csv('./log/attack/{}/attack_log_m{}_z{}.csv'.format(experienment_name, m, z), index=False)
            global_attack_results['ASR'].iloc[m, z] = attack_res[(m,z)]['succ_atk'] / attack_res[(m, z)]['atk']
            global_attack_results['ACC'].iloc[m, z] = accuracy_score(
                df_log['true_label'].tolist(), df_log['adv_label'].tolist(), normalize=True)
            global_attack_results['F1'].iloc[m, z]  = f1_score(
                df_log['true_label'].tolist(), df_log['adv_label'].tolist(), pos_label=1)
    for item in global_attack_results:
            global_attack_results[item].to_csv('./log/attack/{}/{}.csv'.format(experienment_name, item), index=False)
    
file.close()

if opt['steps'] > 0:
    loss_plot('./log/attack', experienment_name, opt['steps'])
    