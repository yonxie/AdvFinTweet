# -*- coding: utf-8 -*-
"""
@file: han
@author: Yong Xie
@time: 8/12/2021 9:37 AM
@Description:

This implementation of model HAN in Hierarchical Attention Networks for Document Classification

"""

import sys, os
sys.path.append('..')
sys.path.append('.')
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
torch.set_default_dtype(torch.float16)
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from model.BaseModel import BaseModel
from model.BaseModelWrapper import BaseWrapper


class WordAttNet(nn.Module):
    def __init__(self, hidden_size, embed_size, device):
        super(WordAttNet, self).__init__()
        
        # self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        # self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        # self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        
        self.att_weight = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.att_context = nn.Linear(2 * hidden_size, 1, bias=False)
        
        self.device = device
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        # self._create_weights(mean=0.0, std=0.05)

    def forward(self, input, mask):
        """input is packed sequence for one sentence"""
        
        f_output, h_output = self.gru(input)  # feature output and hidden state output
        output = self.att_weight(f_output)
        output = self.att_context(output)
        mask = torch.unsqueeze(mask, -1)
        output = F.softmax(output*mask-500*(1-mask), 1)
        output = torch.sum(f_output * output, 1, keepdim=True)

        return output, h_output.to(self.device)
    

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        # self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        # self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        # self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.att_weight = nn.Linear(2 * sent_hidden_size, 2 * sent_hidden_size)
        self.att_context = nn.Linear(2 * sent_hidden_size, 1, bias=False)
        
        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * sent_hidden_size, sent_hidden_size)
        # self._create_weights(mean=0.0, std=0.05)

    # def _create_weights(self, mean=0.0, std=0.05):
    #     self.sent_weight.data.normal_(mean, std)
    #     self.context_weight.data.normal_(mean, std)

    def forward(self, input, mask):
        """representation for each sentence"""
        
        f_output, h_output = self.gru(input)
        output = self.att_weight(f_output)
        output = self.att_context(output)
        mask = torch.unsqueeze(mask, -1)
        output = F.softmax(output*mask-500*(1-mask), 1)
        output = torch.sum(f_output * output, 1)
        output = self.fc(output)

        return output, h_output
    
    
class HierAttNet(BaseModel):
    # def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                #  max_sent_length, max_word_length):
    def __init__(self, ebd_size, fea_size, msg_hdn_size, embedding, device, config, syn_layer=None, max_n_msgs=30):
        super(HierAttNet, self).__init__(embedding=embedding, device=device, config=config, syn_layer=syn_layer)
        
        self.word_hidden_size = ebd_size
        self.sent_hidden_size = msg_hdn_size
        self.max_sent_length = max_n_msgs
        self.max_word_length = config['model']['max_n_words']
        self.feature_size = fea_size
        self.device = device
        self.word_att_net = WordAttNet(self.word_hidden_size, ebd_size, device)
        self.sent_att_net = SentAttNet(self.sent_hidden_size, self.word_hidden_size, self.sent_hidden_size)

        if self.feature_size > 0:
            self.w_tech = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size * 2),
                nn.BatchNorm1d(self.feature_size * 2),
                nn.ReLU(),
                nn.Linear(self.feature_size * 2, self.feature_size),
                nn.BatchNorm1d(self.feature_size),
                nn.ReLU()
            )

        self.dense = nn.Sequential(
            nn.Linear(self.sent_hidden_size+self.feature_size, (self.sent_hidden_size+self.feature_size) // 2),
            nn.BatchNorm1d(num_features=(self.sent_hidden_size+self.feature_size) // 2), 
            nn.ReLU(),
            nn.Linear((self.sent_hidden_size+self.feature_size) // 2, 2)
        )
        
    # def _init_hidden_state(self, batch_size):
    #     self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
    #     self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
    #     if torch.cuda.is_available():
    #         self.word_hidden_state = self.word_hidden_state.cuda()
    #         self.sent_hidden_state = self.sent_hidden_state.cuda()

    def normal_forward(self, inputs, **keywords):

        text, msg_count, msg_mask, word_count, word_mask, technical, y = inputs
        batch_size, nticker, nday, nmsg, nword, embed_dim = text.shape
        
        text = torch.squeeze(text, dim=1)  # (batch size, day, message, words)
        technical = torch.squeeze(technical, dim=1)
        word_count = torch.squeeze(word_count, dim=1)
        word_mask = torch.squeeze(word_mask, dim=1)
        msg_mask = torch.squeeze(msg_mask, dim=1)
        msg_count = torch.squeeze(msg_count, dim=1)
        y = torch.squeeze(y, dim=1)
        
        # self._init_hidden_state(batch_size=batch_size)
        
        tweet_embed_list = []
        attack = text.shape[2] > self.max_sent_length
        if attack:
            tweet_mask = torch.zeros(batch_size, self.max_sent_length * (nday+1)).to(self.device)
        else:
            tweet_mask = torch.zeros(batch_size, self.max_sent_length * nday).to(self.device)
        for t in range(nday):
            for m in range(nmsg):
                if (t < 4 and m >= self.max_sent_length):
                    continue
                else:
                    tweet_mask[:, t * 5 + m] = msg_mask[:, t, m]
                    x_text = text[:, t, m, :, :]
                    token_mask = word_mask[:, t, m].view(batch_size, -1)
                    output, self.word_hidden_state = self.word_att_net(x_text, token_mask)
                    tweet_embed_list.append(output)
        tweet_embed = torch.cat(tweet_embed_list, 1).to(self.device)
        # tweet_mask = msg_mask.view(batch_size, -1)
        text_hdn, _ = self.sent_att_net(tweet_embed, tweet_mask)
        if self.feature_size > 0:
            # tech_feature = self.w_tech(technical.view(batch_size, -1))
            tech_feature = self.w_tech(technical[:,-1,:])
            hdn = torch.cat([text_hdn, tech_feature], dim=1)
        else:
            hdn = text_hdn.clone()
        
        output = self.dense(hdn)    
        output = F.softmax(output, 1)   

        return torch.unsqueeze(output, 1), _, _, _

class HANWrapper(BaseWrapper):
    def __init__(self, name, network, optimizer, votes, cfg):
        super(HANWrapper, self).__init__(name=name, network=network, optimizer=optimizer, votes=votes, cfg=cfg)
        
        self.optimizer = 'SGD'
        if self.optimizer == 'Adam':
            # optim = torch.optim.Adam(params=self.network.parameters(), lr=self.cfg['train']['lr'], eps=self.cfg['train']['epsilon'])
            self.optim = torch.optim.Adam(params=self.network.parameters(), 
                    lr=self.cfg['train']['lr'], weight_decay=self.cfg['train']['weight_decay'])
        elif self.optimizer == 'SGD':
            self.cfg['train']['momentum'] = 0
            self.optim = torch.optim.SGD(params=self.network.parameters(),
                                    lr=self.cfg['train']['lr'], momentum=self.cfg['train']['momentum'])
        else:
            raise Exception('the optimizer is not supported')

        self.scheduler = StepLR(self.optim, 
                step_size=self.cfg['train']['schedule_step'], 
                gamma=self.cfg['train']['schedule_gamma'])
    
    
if __name__ == '__main__':
    from model.config_loader import config, path
    from model.utils import GloveEmbedding
    from model.data_loader import StockData
    from torch.utils.data import DataLoader
    
    device = 'cuda'
    
    embedding = GloveEmbedding(resource=path.glove, n_d=50, vocab_list=config['vocab']).to(device)
    network = HierAttNet(ebd_size=50, fea_size=config['model']['tech_feature_size'], 
                    msg_hdn_size=50, embedding=embedding, device=device, config=config).to(device).double()

    test_dset = StockData(config['dates']['test_start_date'], config['dates']['test_end_date'], ['AAPL'], device)
    test_loader = DataLoader(test_dset, 32, shuffle=False)
    
    for i, inputs in enumerate(test_loader, 0):
        if i > 0:
            continue
        print(i)
        network(inputs, 1)