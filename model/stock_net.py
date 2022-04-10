# -*- coding: utf-8 -*-
"""
@file: stock_net
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

from torch.autograd import Variable
from collections import defaultdict

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
        self.w1 = nn.Linear(input_size, input_size, bias=False)
        self.w2 = nn.Linear(input_size, 1, bias=False)

    def forward(self, x, msg_mask):
        """
        :param x: massage hidden, (batch, message, message hidden)
        :param msg_mask: message mask (batch, message)
        """
        # x = self.dropout_ce(x)
        att_x = torch.squeeze(self.w2(F.tanh(self.w1(x))), dim=2)
        u = F.softmax(att_x - 500 * (1 - msg_mask), 1)
        return torch.sum(x * torch.unsqueeze(u, -1), dim=1)

class MarketEncoderCell(nn.Module):
    """Market Information Encoder"""

    def __init__(self, input_size, h_s_size, h_z_size, z_size, y_size, dpt_in):
        super(MarketEncoderCell, self).__init__()
        self.input_size = input_size

        self.dropout = nn.Dropout(p=dpt_in)    
        self.GRU = nn.GRUCell(input_size=input_size, hidden_size=h_s_size)
        
        self.w_hz_post = nn.Sequential(
                nn.Linear(z_size+input_size+h_s_size+y_size, h_z_size),
                nn.BatchNorm1d(h_z_size, affine=False, track_running_stats=False),
                nn.Tanh()
        )
        self.sigma_post = nn.Linear(h_z_size, z_size)
        self.mu_post = nn.Linear(h_z_size, z_size)

        self.sigma_prior = nn.Linear(h_z_size, z_size)
        self.mu_prior = nn.Linear(h_z_size, z_size)

        self.w_hz_prior = nn.Sequential(
                nn.Linear(z_size+input_size+h_s_size, h_z_size),
                nn.BatchNorm1d(h_z_size, affine=False, track_running_stats=False),
                nn.Tanh()
        )

    def forward(self, x, h, z, y=None):

        # GRU
        # x = self.dropout(x)
        h_s = self.GRU(x, h)
        
        if y != None:
            # calculate posterior
            h_z_post = self.w_hz_post(torch.cat([z, x, h_s, y.half()], dim=1))
            u_post = self.mu_post(h_z_post)
            logvar_post = self.sigma_post(h_z_post)
        else:
            u_post, logvar_post = None, None
            
        # calculate prior
        h_z_prior = self.w_hz_prior(torch.cat([z, x, h_s], dim=1))
        u_prior = self.mu_prior(h_z_prior)
        logvar_prior = self.sigma_prior(h_z_prior)

        return u_prior, logvar_prior, u_post, logvar_post, h_s


class AttentiveTemporalAuxiliary(nn.Module):
    def __init__(self, g_size, device):
        super(AttentiveTemporalAuxiliary, self).__init__()

        self.w_gi = nn.Sequential(
            nn.Linear(g_size, g_size, bias=False),
            nn.BatchNorm1d(g_size, affine=False, track_running_stats=False),
            nn.Tanh()
        )
        self.w_i = nn.Linear(g_size, 1)
        self.w_gd = nn.Linear(g_size, g_size)
        self.device = device

        self.y_T = nn.Sequential(
            nn.Linear(g_size+2, 2),
            nn.Softmax()
        )

    def forward(self, G, Y, alpha):
        vi = self.w_i(F.tanh(self.w_gi(G[:, :-1, :])))
        vd = F.tanh(self.w_gd(G[:, :-1, :])).matmul(torch.unsqueeze(G[:, -1, :], -1))
        v = F.softmax(vi * vd, 1)
        v_ = v.permute([0, 2, 1])
        wi_y = torch.squeeze(v_.matmul(Y[:, :-1, :]), dim=-2)
        wi_y = torch.cat([wi_y, torch.squeeze(G[:, -1, :], -1)], dim=1)
        y_T = torch.unsqueeze(self.y_T(wi_y), dim=1)
        return y_T, torch.cat([v * alpha, torch.ones([v.shape[0], 1, 1], device=self.device, requires_grad=True)], dim=1)


class StockNet(BaseModel):
    """
    # StockNet in Pytorch
    """
    def __init__(self, ebd_size, fea_size, msg_hdn_size, embedding, device, config, syn_layer=None):
        super(StockNet, self).__init__(embedding=embedding, device=device, config=config, syn_layer=syn_layer)
        # super(StockNet, self).__init__()
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
        self.msg_GRU = nn.Sequential(
            nn.GRU(input_size=ebd_size, hidden_size=msg_hdn_size, num_layers=1, bidirectional=True),
            # nn.Dropout(p=config['model']['dropout_mel'])
        )
        self.msg_attention = MessageAttention(msg_hdn_size, config['model']['dropout_ce'])
        self.vmd = MarketEncoderCell(msg_hdn_size+fea_size, h_s_size=self.hs_size,
                                     h_z_size=self.z_size, z_size=self.z_size, y_size=1, 
                                     dpt_in=config['model']['dropout_vmd_in'])
        self.w_g = nn.Sequential(
            # nn.Linear(self.hs_size + msg_hdn_size + fea_sze + self.z_size, self.g_size),
            nn.Linear(self.hs_size  + self.z_size, self.g_size),
            nn.BatchNorm1d(self.g_size, affine=False, track_running_stats=False),
            nn.Tanh()
        )
        
        self.w_y_tilde = nn.Sequential(
            nn.Linear(self.g_size, 2),
            nn.Softmax()
        )
        
        if self.feature_size > 0:
            self.w_tech = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size * 2),
                nn.BatchNorm1d(self.feature_size * 2, affine=False, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(self.feature_size * 2, self.feature_size)
            )
        self.hdn_linear = nn.Sequential(
            nn.Linear(self.msg_hdn_size+self.feature_size, self.msg_hdn_size+self.feature_size),
            nn.BatchNorm1d(self.msg_hdn_size+self.feature_size, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        
        # ATA
        self.ata = AttentiveTemporalAuxiliary(self.g_size, self.device)

        self.gen_kl_lambda = self.kl_lambda()
        self.step = torch.tensor(1.0, requires_grad=False).detach()
        self.alpha = torch.tensor(config['model']['alpha'], requires_grad=False, device=device).detach()

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

        h_s = torch.zeros([batch_size, self.hs_size], device=self.device)
        z_post = torch.zeros([batch_size, self.z_size], device=self.device)
        Gs, Y_tildes = [], []
        kls = []
        msg_ebds = []
        for t in range(nday):
            msg_ebd_list = []
            for mm in range(nmsg):
                x_text = text[:, t, mm, :, :]        
                # x_text = self.dropout_mel_in(x_text)
                x_text = pack_padded_sequence(x_text, lengths=word_count[:, t, mm].cpu(),
                                              batch_first=True, enforce_sorted=False)  # only pack-padded each message
                _, gru_hdn = self.msg_GRU(x_text)  # message level embedding
                gru_hdn = gru_hdn.view(2, batch_size, self.msg_hdn_size)
                msg_ebd_list.append(torch.mean(gru_hdn, [0]).view(batch_size, 1, -1))
            msg_ebd = torch.cat(msg_ebd_list, dim=1)
            msg_ebds.append(msg_ebd)
        # combine messages for each day (batch size, hidden size)
        for t in range(nday):               
            # concat numerical features
            text_hdn = self.msg_attention(msg_ebds[t].to(self.device), msg_mask[:, t, :])
            if self.feature_size > 0:
                tech_feature = self.w_tech(technical[:, t, :])
                hdn = torch.cat([text_hdn, tech_feature], dim=1)
            else:
                hdn = text_hdn.clone()
            
            hdn = self.hdn_linear(hdn)  
            
            # encoder
            # if not self.training and t == (nday-1):
            
            if  t == (nday-1):
                u_prior, logvar_prior, _, _, h_s = self.vmd(hdn, h_s, z_post, y=None)
                z_prior = u_prior
                # g = self.w_g(torch.cat([hdn, h_s, z_prior], dim=1))
                g = self.w_g(torch.cat([h_s, z_prior], dim=1))
                y_tilde = self.w_y_tilde(g)
                
                # G[:, t, :], Y_tilde[:, t, :] = g, y_tilde
                Gs.append(torch.unsqueeze(g,1))
                Y_tildes.append(torch.unsqueeze(y_tilde,1))
            else:
                u_prior, logvar_prior, u_post, logvar_post, h_s = self.vmd(hdn, h_s, z_post, y[:, [t]])
                kl = self._normal_kl_div([u_post, logvar_post], [u_prior, logvar_prior])
                kls.append(torch.sum(kl, dim=1).view(batch_size, 1))
                z_prior = u_prior          
                # z_post = u_post + torch.sqrt(torch.exp(logvar_post)) * torch.normal(0, 1, [batch_size, self.z_size], device=self.device)
                z_post = u_post
                # g = self.w_g(torch.cat([hdn, h_s, z_post], dim=1))
                g = self.w_g(torch.cat([h_s, z_post], dim=1))
                y_tilde = self.w_y_tilde(g)
                # G[:, t, :], Y_tilde[:, t, :] = g, y_tilde
                Gs.append(torch.unsqueeze(g,1))
                Y_tildes.append(torch.unsqueeze(y_tilde,1))
        Y_tilde = torch.cat(Y_tildes, dim=1)
        G = torch.cat(Gs, dim=1)  
              
        # decoder
        # Y_tilde[:, -1, :]v = self.ata(G, Y_tilde, self.alpha)
        Y_tildes[-1], v = self.ata(G, Y_tilde, self.alpha)
        y_tilde = torch.cat(Y_tildes, dim=1)
        Y_tilde = torch.squeeze(Y_tilde, dim=-1)
        v = torch.squeeze(v, dim=-1)
        loss, mle, kl = self.calc_loss(Y_tilde, y, v, kls)
        return Y_tilde, loss, mle, kl

    def calc_loss(self, y_p, y, v, kls):
        """Calculate MLE loss + KL loss."""

        nday = y.shape[1]
        mle_loss = []
        for t in range(nday):
            mle = F.nll_loss(torch.log(y_p[:, t, :]), y[:, t].long(), reduction='none')
            mle_loss.append((v[:, t] * mle).view(mle.shape[0], -1))
        mle_loss = torch.mean(torch.sum(torch.cat(mle_loss, dim=1), dim=1))
        kls = torch.cat(kls, dim=1)
        if self.training:
            kl_lambda = next(self.gen_kl_lambda) 
        else:
            kl_lambda = min(self.step * self.config['model']['kl_lambda_aneal_rate'], 1.0)
        if kls.shape[1] == v.shape[1]:
            kl_loss = torch.mean(torch.sum(kls * v, dim=1)) * kl_lambda
        else:
            kl_loss = torch.mean(torch.sum(kls * v[:,:-1], dim=1)) * kl_lambda 
        return kl_loss+mle_loss, mle_loss, kl_loss / kl_lambda

    def kl_lambda(self):
        """calculate lambda"""
        
        while True:
            # self.step += torch.tensor(0.01).detach()
            # self.step += torch.tensor(1).detach()
            yield min(self.step * self.config['model']['kl_lambda_aneal_rate'], 1.0)

    @staticmethod
    def _normal_kl_div(input, target):
        """
        # Calculate KL divergence of two normal distribution.

        # Args:
            input: [mean: (N, *), logvar: (N, *)]
            target: [mean: (N, *), logvar: (N, *)
        """

        part1 = (target[1] - input[1]) / 2
        part2 = (torch.exp(input[1]) + torch.pow(input[0] - target[0], 2)) / 2 / torch.exp(target[1])
        
        return part1 + part2 - 1/2

    def get_input_embeddings(self):
        """Query embedding layers"""
        return self.embedding.embedding
    
    
class StocknetWrapper(BaseWrapper):
    def __init__(self, name, network, optimizer, votes, cfg):
        super(StocknetWrapper, self).__init__(name=name, network=network, optimizer=optimizer, votes=votes, cfg=cfg)
        
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
