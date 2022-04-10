# -*- coding: utf-8 -*-
"""
@file: config_loader
@author: Yong Xie
@time: 5/12/2021 11:18 AM
@Description:

# this script load configurations.

"""

import logging
import logging.config
import os
import io, re
import json
import sys
import warnings
import argparse
sys.path.append("..") 

warnings.filterwarnings("ignore")

def parse_config():

    parser = argparse.ArgumentParser()
    
    # path argument
    parser.add_argument('--paths_data', type=str, default='data/', help='relative data path')   
    parser.add_argument('--paths_text', type=str, default='text/tweet', help='relative text path')
    parser.add_argument('--paths_label', type=str, default='label/return', help='relative label path')
    parser.add_argument('--paths_technical', type=str, default='technical', help='relatice technical feature path')
    parser.add_argument('--paths_resource', type=str, default='./resource', help='absolute resource path')
    parser.add_argument('--paths_glove', type=str, default='glove.twitter.27B.50d.txt', help='glove file name')
    parser.add_argument('--paths_vocab_tweet', type=str, default='extended_customized_vocab_fin_tweet.txt', help='vocal file name')
    parser.add_argument('--paths_syn_tweet', type=str, default='syn_extended_customized_fin_tweet_15.csv', help='synonyms file name')
    parser.add_argument('--paths_checkpoints', type=str, default='checkpoints', help='relative checkpoints folder path')
    parser.add_argument('--paths_log', type=str, default='log/', help='relative log folder')
    parser.add_argument('--paths_graphs', type=str, default='graphs', help='relative graph folder')
    
    # model argument 
    parser.add_argument('--model_name', type=str, default='tweetgru', help='name of model to be used')  
    parser.add_argument('--model_word_embed_type', type=str, default='glove', help='word embedding type')
    parser.add_argument('--model_weight_init', type=str, default='xavier_uniform', help='weight initialiation method')
    parser.add_argument('--model_mel_cell_type', type=str, default='gru', help='MEL module cell type')
    parser.add_argument('--model_vmd_cell_type', type=str, default='gru', help='market decoder cell type')
    
    parser.add_argument('--model_word_embed_size', type=int, default=50, help='embedding size')
    parser.add_argument('--model_stock_embed_size', type=int, default=150, help='')
    parser.add_argument('--model_tech_feature_size', type=int, default=3, help='')
    parser.add_argument('--model_init_stock_with_word', type=int, default=0, help='')
    parser.add_argument('--model_mel_h_size', type=int, default=100, help='')
    parser.add_argument('--model_h_size', type=int, default=150, help='')
    parser.add_argument('--model_g_size', type=int, default=100, help='')
    parser.add_argument('--model_alpha', type=float, default=0.5, help='eights for previous date')
    parser.add_argument('--model_dropout_mel_in', type=float, default=0.3, help='')
    parser.add_argument('--model_drpoout_ce', type=float, default=0.1, help='')
    parser.add_argument('--model_dropout_vmd_in', type=float, default=0.3, help='')
    parser.add_argument('--model_dropout_ce', type=float, default=0.0, help='')
    parser.add_argument('--model_dropout_vmd', type=float, default=0.0, help='')
    parser.add_argument('--model_kl_lambda_aneal_rate', type=float, default=0.0075, help='')
    parser.add_argument('--model_device', type=str, default='cuda')

    parser.add_argument('--model_day_step', type=int, default=5, help='')
    parser.add_argument('--model_day_shuffle', type=int, default=0, help='')
    parser.add_argument('--model_ticker_combine', type=int, default=0, help='')
    parser.add_argument('--model_ticker_shuffle', type=int, default=0, help='')
    parser.add_argument('--model_text_combine', type=int, default=0, help='')
    parser.add_argument('--model_max_n_msgs', type=int, default=30, help='')
    parser.add_argument('--model_max_n_words', type=int, default=40, help='')
    parser.add_argument('--model_threshold', type=int, default=1, help='')
    
    parser.add_argument('--model_prefix', type=str, default='', help='')
    
    # train argument
    parser.add_argument('--train_lr', type=float, default=0.005, help='')
    parser.add_argument('--train_epsilon', type=int, default=1, help='')
    parser.add_argument('--train_epochs', type=int, default=30, help='')
    parser.add_argument('--train_min_date_ratio', type=int, default=0, help='')
    parser.add_argument('--train_weight_decay', type=float, default=0.01, help='')
    parser.add_argument('--train_schedule_step', type=int, default=5, help='')
    parser.add_argument('--train_schedule_gamma', type=float, default=0.5, help='')
    parser.add_argument('--train_resume', type=int, default=0, help='')
     
    # stocks argument
    parser.add_argument('--stocks_materials', type=list, default=['XOM', 'RDS-B', 'PTR', 'CVX', 'TOT', 'BP', 'BHP', 'SNP', 'SLB', 'BBL'], help='')
    parser.add_argument('--stocks_consumer_goods', type=list, default=['PG', 'BUD', 'KO', 'PM', 'TM', 'PEP', 'UN', 'UL', 'MO'], help='')
    parser.add_argument('--stocks_healthcare', type=list, default=['JNJ', 'PFE', 'NVS', 'UNH', 'MRK', 'AMGN', 'MDT', 'ABBV', 'SNY', 'CELG'], help='')
    parser.add_argument('--stocks_services', type=list, default=['AMZN', 'BABA', 'WMT', 'CMCSA', 'HD', 'DIS', 'MCD', 'CHTR', 'UPS', 'PCLN'], help='')
    parser.add_argument('--stocks_utilities', type=list, default=['NEE', 'DUK', 'D', 'SO', 'NGG', 'AEP', 'PCG', 'EXC', 'SRE', 'PPL'], help='')
    parser.add_argument('--stocks_cong', type=list, default=['IEP', 'HRG', 'CODI', 'REX', 'SPLP', 'PICO', 'AGFS', 'GMRE'], help='')
    parser.add_argument('--stocks_finance', type=list, default=['BCH', 'BSAC', 'BRK-A', 'JPM', 'WFC', 'BAC', 'V', 'C', 'HSBC', 'MA'], help='')
    parser.add_argument('--stocks_industrial_goods', type=list, default=['GE', 'MMM', 'BA', 'HON', 'UTX', 'LMT', 'CAT', 'GD', 'DHR', 'ABB'], help='')
    parser.add_argument('--stocks_tech', type=list, default=['GOOG', 'MSFT', 'FB', 'T', 'CHL', 'ORCL', 'TSM', 'VZ', 'INTC', 'CSCO'], help='')
    
    parser.add_argument('--dates_train', type=str, 
                        default=[
                            ['2014-01-01', '2014-03-31'],
                            ['2014-04-16', '2014-07-15'],
                            ['2014-08-01', '2014-10-31'],
                            ['2014-11-16', '2015-02-15'],
                            ['2015-03-01', '2015-05-30'],
                            ['2015-06-16', '2015-09-15'],
                            ['2015-10-01', '2015-12-15']],
                        help='')
    parser.add_argument('--dates_test', type=list, 
                        default=[
                            ['2014-04-01', '2014-04-15'], 
                            ['2014-07-16', '2014-07-31'],
                            ['2014-11-01', '2014-11-15'],
                            ['2015-02-16', '2015-02-28'],
                            ['2015-06-01', '2015-06-15'],
                            ['2015-09-16', '2015-09-30'],  
                            ['2015-12-16', '2015-12-31']], help='')
    
    # stage argument
    parser.add_argument('--stage', type=str, default='train', help='')
    
    # PGD argument
    parser.add_argument('--pgd_optim', type=str, default='adam', help='')
    parser.add_argument('--pgd_weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--pgd_atk_days', type=list, default=[-1], help='')
    parser.add_argument('--pgd_steps', type=int, default=15, help='')
    parser.add_argument('--pgd_lr', type=float, default=1, help='')
    parser.add_argument('--pgd_smooth', type=int, default=0, help='')
    parser.add_argument('--pgd_alt', type=int, default=0, help='')
    parser.add_argument('--pgd_sparsity', type=float, default=0, help='')
    parser.add_argument('--pgd_prefix', type=str, default='', help='')
    parser.add_argument('--pgd_schedule_step', type=int, default=0, help='')
    parser.add_argument('--pgd_schedule_gamma', type=float, default=0.5, help='')
    parser.add_argument('--pgd_projection', type=int, default=1, help='')
    parser.add_argument('--pgd_fix', type=int, default=0, help='')
    parser.add_argument('--pgd_m_num', type=int, default=1, help='')
    parser.add_argument('--pgd_z_num', type=int, default=1, help='')
    parser.add_argument('--pgd_r_num', type=int, default=1, help='')
    parser.add_argument('--pgd_train_name', type=str, default='', help='')
    parser.add_argument('--pgd_model_name', type=str, default='', help='')
    parser.add_argument('--pgd_ckpt_epoch', type=int, default=1, help='')
    parser.add_argument('--pgd_concat', type=int, default=1, help='')
    parser.add_argument('--pgd_batch', type=int, default=32, help='')
    parser.add_argument('--pgd_attack_type', type=str, default='replacement', help='attack methods: replacement/deletion')
    parser.add_argument('--pgd_opt_method', type=str, default='joint', help='optimization methods: joint/greedy')
    parser.add_argument('--pgd_attack_corpus', type=str, default='all', help='optimization methods: joint/greedy')
    
    # parse argument
    args = parser.parse_args()
    
    category = ['paths', 'model', 'train', 'stocks', 'dates', 'pgd']
    config = {}
    for c in category:
        regex = '^{}_'.format(c)
        config[c] = {}
        length = len(c)
        for arg in args.__dict__.keys():
            if re.findall(regex, arg):
                config[c][arg[length+1:]] = args.__dict__[arg]
    config['stage'] = args.stage


    return config
class PathParser:

    def __init__(self, config_path):
        self.root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir))
        self.log = os.path.join(self.root, config_path['log'])
        self.data = os.path.join(self.root, config_path['data'])
        self.res = config_path['resource']
        self.graphs = os.path.join(self.root, config_path['graphs'])
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])

        self.glove = os.path.join(self.res, config_path['glove'])

        self.technical = os.path.join(self.data, config_path['technical'])
        self.text = os.path.join(self.data, config_path['text'])
        self.label = os.path.join(self.data, config_path['label'])
        self.vocab = os.path.join(self.res, config_path['vocab_tweet'])
        self.syn = os.path.join(self.res, config_path['syn_tweet'])
        self.tickers= os.path.join(self.res, 'tickers.csv')

config = parse_config()  
path = PathParser(config_path=config['paths'])

with io.open(str(path.vocab), 'r', encoding='utf-8') as vocab_f:
    config['vocab'] = json.load(vocab_f)
    config['vocab_size'] = len(config['vocab']) + 1  # for UNK

# logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = os.path.join(path.log, '{0}.log'.format('model'))
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# root = Path(__file__).parent.parent

# with open(os.path.join(root, 'model', 'config.yml'), 'r') as f:
#     config = yaml.load(f.read())

# path = PathParser(config_path=config['paths'])

# with io.open(str(path.vocab), 'r', encoding='utf-8') as vocab_f:
#     config['vocab'] = json.load(vocab_f)
#     config['vocab_size'] = len(config['vocab']) + 1  # for UNK

# # logger
# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG)
# log_fp = os.path.join(path.log, '{0}.log'.format('model'))
# file_handler = logging.FileHandler(log_fp)
# console_handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)


if __name__ == '__main__':
    parse_config()