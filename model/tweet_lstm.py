# -*- coding: utf-8 -*-
"""
@file: tweet_lstm
@author: Yong Xie
@time: 5/13/2021 3:37 PM
@Description:

This script replicates StockNets

"""

# from pgd.gradient_attack import *
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.set_default_dtype(torch.float16)
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
# from pgd.utils import *
from model.BaseModel import BaseModel
from model.BaseModelWrapper import BaseWrapper


class MessageAttention(torch.nn.Module):
    """
    # Message level attention model
    """

    def __init__(self, input_size, dpt_in):
        super(MessageAttention, self).__init__()
        self.input_size = input_size
        self.dropout_ce = nn.Dropout(p=dpt_in)
        self.w = nn.Sequential(
            nn.Linear(input_size, input_size, bias=False),
            nn.Tanh(),
            nn.Linear(input_size, 1, bias=False)
        )

    def forward(self, x, msg_mask):
        """
        :param x: massage hidden, (batch, message, message hidden)
        :param msg_mask: message mask (batch, message)
        """
        att_x = torch.squeeze(self.w(x), dim=2)
        u = F.softmax(att_x - 500 * (1 - msg_mask), 1)
        return torch.sum(x * torch.unsqueeze(u, -1), dim=1)

class TweetLSTM(BaseModel):
    """
    # TweetLSTM in Pytorch
    """
    def __init__(self, ebd_size, fea_size, msg_hdn_size, embedding, device, config, syn_layer=None):
        super(TweetLSTM, self).__init__(embedding=embedding, device=device, config=config, syn_layer=syn_layer)
        self.msg_hdn_size = msg_hdn_size
        self.feature_size = fea_size
        self.z_size = 150
        self.hs_size = 100
        self.g_size = 50
        self.device = device
        self.config = config
        self.embedding = embedding
        self.embedding.requires_grad_(False)
        
        # Networks
        self.dropout_mel_in = nn.Dropout(p=config['model']['dropout_mel_in'])
        self.msg_LSTM = nn.Sequential(
            nn.LSTM(input_size=ebd_size, hidden_size=msg_hdn_size, 
                   num_layers=1, bidirectional=True, batch_first=True),
        )
        self.msg_attention = MessageAttention(msg_hdn_size, config['model']['dropout_ce'])
        
        if self.feature_size > 0:
            self.w_tech = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size * 2),
                nn.BatchNorm1d(self.feature_size * 2),
                nn.ReLU(),
                nn.Linear(self.feature_size * 2, self.feature_size),
                nn.BatchNorm1d(self.feature_size),
                nn.ReLU()
            )

        self.day_LSTM = nn.LSTM(input_size=self.feature_size+self.msg_hdn_size, num_layers=1,
                              hidden_size=msg_hdn_size, bidirectional=False, batch_first=True)
        
        self.dense = nn.Sequential(
            nn.Linear(self.msg_hdn_size, msg_hdn_size // 2),
            nn.BatchNorm1d(num_features=msg_hdn_size // 2), 
            nn.ReLU(),
            nn.Linear(self.msg_hdn_size // 2, 2)
        )

    def normal_forward(self, inputs, **keywords):
    
        # vocab_size = self.embedding.embedding.weight.data.shape[0]
        text, msg_count, msg_mask, word_count, word_mask, technical, y = inputs
        # text = self.embedding(text.int())
        batch_size, nticker, nday, nmsg, nword, embed_dim = text.shape

        text = torch.squeeze(text, dim=1)  # (batch size, day, message, words)
        technical = torch.squeeze(technical, dim=1)
        word_count = torch.squeeze(word_count, dim=1)
        msg_mask = torch.squeeze(msg_mask, dim=1)
        msg_count = torch.squeeze(msg_count, dim=1)
        y = torch.squeeze(y, dim=1)

        hdns = []
        
        # obatain tweet representation
        for t in range(nday):
            msg_ebd_list = []
            for mm in range(nmsg):
                x_text = text[:, t, mm, :, :]        
                x_text = self.dropout_mel_in(x_text)
                x_text = pack_padded_sequence(x_text, lengths=word_count[:, t, mm].cpu(),
                                              batch_first=True, enforce_sorted=False)  # only pack-padded each message
                _, (gru_hdn, _) = self.msg_LSTM(x_text)  # message level embedding
                gru_hdn = gru_hdn.view(2, batch_size, self.msg_hdn_size)
                msg_ebd_list.append(torch.mean(gru_hdn, [0]).view(batch_size, 1, -1))
            msg_ebd = torch.cat(msg_ebd_list, dim=1)
             
            # concat numerical features
            text_hdn = self.msg_attention(msg_ebd.to(self.device), msg_mask[:, t, :])
            if self.feature_size > 0:
                tech_feature = self.w_tech(technical[:, t, :])
                hdn = torch.cat([text_hdn, tech_feature], dim=1)
            else:
                hdn = text_hdn.clone()
            hdns.append(torch.unsqueeze(hdn, 1))
        
        fea_hdn = torch.cat(hdns, dim=1)
        _, (fea_hdn, _) = self.day_LSTM(fea_hdn)
        fea_hdn = torch.squeeze(fea_hdn, 0)
        output = self.dense(fea_hdn)
        output = F.softmax(output, 1)
        return torch.unsqueeze(output, 1), _, _, _

    def get_input_embeddings(self):
        """Query embedding layers"""
        return self.embedding.embedding
    
    
class TweetLSTMWrapper(BaseWrapper):
    def __init__(self, name, network, optimizer, votes, cfg):
        super(TweetLSTMWrapper, self).__init__(name=name, network=network, optimizer=optimizer, votes=votes, cfg=cfg)
        
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




