# -*- coding: utf-8 -*-
"""
@file: BaseModel.py
@author: Yong Xie
@time: 6/17/2021 11:18 AM
@Description:

This script houses base model class of adversial attack. 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable, backward
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import sys
sys.path.append('..')
sys.path.append('.')
from model.utils import *
from model.data_loader import StockData
from model.config_loader import path, config, logger


torch.set_default_dtype(torch.double)

class BaseModel(nn.Module):
    def __init__(self, embedding, device, syn_layer=None, config=None):
        super(BaseModel, self).__init__()
        self.embedding = embedding
        self.config = config
        self.device = device
        self.syn_layer = syn_layer
        
    def normal_forward(self, inputs):
        """need to be overwritted by child class."""
        pass
    
    def forward(self, inputs, attack=None, mode='normal_forward', **kwargs):
        """forward propagation"""
        if mode == 'normal_forward':
            # inputs[0] = self.embedding(inputs[0].long())
            # return self.normal_forward(inputs)
            text_embd = self.embedding(inputs[0].long())
            return self.normal_forward([text_embd]+inputs[1:], **kwargs)
        if mode == 'pgd_forward':
            return self.pgd_forward(inputs, attack, **kwargs)
        if mode == 'pgd_joint_attack':
            return self.pgd_joint_attack(inputs, attack, **kwargs)
        if mode == 'pgd_greedy_attack':
            return self.pgd_greedy_attack(inputs, attack, **kwargs)
        else:
            raise Exception('found unexpeced model mode: {}'.format(mode))
        
    def pgd_forward(self, inputs, attack, **kwargs):
        """Forward propagation with attack input, 
        no optimization involved.

        Args:
            inputs (list): 
            attack (dict): dict attack parameters: [opt, mw, zw, rw, syn, onehot]
        """
        opt = self.config['pgd']
        
        # assert keys
        for key in ['opt', 'mw', 'zw', 'rw', 'syn', 'onehot']:
            assert key in attack
        
        # m and z always onehot   
        if attack['onehot']:
            _, m_idx = torch.topk(attack['mw'].float(), opt['tweet_k'], dim=-1)
            _, z_idx = torch.topk(attack['zw'].float(), opt['work_k'], dim=-1)
            _, r_idx = torch.topk(attack['rw'].float(), 1, dim=-1)
    
    def _major_vote(self, inputs, votes=1):
        """majority vote"""
  
        if inputs[0].dim() == 6:
            return self.normal_forward(inputs, votes=votes)
        elif inputs[0].dim() == 5:
            return self.forward(inputs, votes=votes)
    
    def pgd_attack_textfooler_test(self, inputs, attack, **kwargs):
        """find the optimized attack by alternating optimization
        """
        
        if 'votes' in kwargs:
            votes = kwargs['votes']
        else:
            votes = 1
                        
        opt = self.config['pgd']
        steps = opt['steps']
        optm_lr = opt['lr']
        init = 'textfooler'   
        assert self.syn_layer != None
        
        orig_text = inputs[0].detach().clone().int()
        inputs[0] = self.embedding(orig_text)     # trasnform text id to 

        # find syns
        syn_id, syn_mask = self.syn_layer(orig_text)
        syn_embd = self.embedding(syn_id)
        
        if len(syn_embd.shape) - len(syn_mask.shape) == 1:
            syn_mask.unsqueeze_(-1)
        
        batch_size, nticker, nday, nmsg, nword, embed_dim = inputs[0].shape
        batch_size, nticker, nday, nmsg, nword, num_syn, embed_dim = syn_embd.shape 
    
        # initialize attack variable
        m_dis = torch.zeros(batch_size, nticker, nday, nmsg, 1, 1, device=self.device)
        z_dis = torch.zeros(batch_size, nticker, nday, nmsg, nword, 1, device=self.device)
        r_dis = torch.zeros(batch_size, nticker, nday, nmsg, nword, num_syn, 1,  device=self.device)   
        nn.init.kaiming_normal_(m_dis), nn.init.kaiming_normal_(z_dis), nn.init.kaiming_normal_(r_dis)
        
        # m_dis = topk_scatter(m_dis, 1, -3, self.device)
        # z_dis = topk_scatter(z_dis, 1, -2, self.device)
        # r_dis = topk_scatter(r_dis, 1, -2, self.device)
        
        orig_pred_ = self._major_vote(inputs)[0]
        orig_label = orig_pred_[:,-1,:].argmax(-1)
        true_label = inputs[-1][:, 0, -1]
        orig_correct = orig_label == true_label
        
        if torch.sum(orig_correct) == 0:
            return batch_size, 0, 0, 0
        
        first_corr = 0
        for idx in range(batch_size):
            if orig_correct[idx] == 1:
                break
            else:
                first_corr += 1
                            
        notation_set = set(['<number>', '<percent>', '<ticker>', '<email>', '<money>', 
                        '<phone>', '<user>', '<time>', '<date>', 'UNK', 'PAD'])
        symbol_set = set(['&', '<', '>', 'r', 'n'])
        avoid_set = notation_set.union(symbol_set)
        textfoolor_succ = 0
        replacements = []
        for i in range(orig_correct.shape[0]):
            
            if not orig_correct[i]:
                continue
            if init == 'textfooler':
                replace = []
                # replace = textfoolor([orig_text[[i]]]+[input[[i]] for input in inputs][1:], true_label[i].cpu().item(),
                                    # self._major_vote, {}, self.embedding, self.syn_layer, 
                                    # attack['use'], -1, 0.5, 15, 10, avoid=avoid_set, 
                                    # top_msgs=3, top_words=3, batch_size=32)[8]
                
            if not replace:
                replace = []
                # continue
            else:
                replacements.append([i]+[replace])
            replace = [(0,0), 0]
            # textfoolor_succ += 1
            
            for r in replace:
                r[1] = self.embedding.word2index(r[1])
                
                r[1] = (syn_id[i, 0, -1, r[0][0], r[0][1]] == r[1]).nonzero(as_tuple=True)[0].cpu().item()
            
                m_dis.data[i, 0, -1, r[0][0]] = 1.0
                z_dis.data[i, 0, -1, r[0][0], r[0][1]] = 1.0
                r_dis.data[i, 0, -1, r[0][0], r[0][1], r[1]] = 1.0
            
            replace = None
        adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], m_dis, z_dis, r_dis, syn_embd, syn_mask, msoft=False, rsoft=False, concat=opt['concat'])
        
        # print('------testfooler inputs----------')
        adv_pred_ = self._major_vote(adv_inputs)[0]
        adv_label = adv_pred_[:, -1, :].argmax(-1)
        
        adv_wrong = adv_label != true_label
        succ_num = torch.sum(adv_wrong * orig_correct).cpu().item()
        atk_num = torch.sum(orig_correct).cpu().item()
        
        perseve = torch.sum(adv_label == orig_label)
        
        # print('perseve rate: {:.3f}'.format(perseve.cpu().item()/batch_size))
            
        return batch_size, atk_num, succ_num, textfoolor_succ
    
    def _textfooler_input(self, input, replacements, syn_ebd):
        
        changed_input, messages = [], []

        for i in range(len(input)):
            item_to_change = input[i].detach().clone()
            # modify text
            if i == 0:
                for b, replacement in replacements:
                    offset = 0
                    for idx, syn_idx in replacement:
                        msg_n = input[1][b, 0, -1].long()
                        tar_msg = item_to_change[b, 0, -1, idx[0]].detach().clone()
                        word_n = input[3][b, 0, -1, idx[0]]
                        tar_msg[idx[1]] = syn_ebd[b, 0, -1, idx[0], idx[1], syn_idx].clone()
                        item_to_change[b, 0, -1, msg_n+offset, :] = tar_msg
                        offset += 1
                
            # modify message count
            elif i == 1:
                for b, replacement in replacements:
                    for idx, syn_idx in replacement:
                        item_to_change[b, 0, -1] += 1
                        
            # modify mask
            elif i == 2:
                for b, replacement in replacements:
                    offset = 0
                    msg_n = input[1][b, 0, -1].long()
                    for idx, syn_idx in replacement:
                        item_to_change[b, 0, -1, msg_n+offset] = 1
                        offset += 1
                        
            # modify word count
            elif i == 3:
                for b, replacement in replacements:
                    offset = 0
                    for idx, syn_idx in replacement:
                        word_n = item_to_change[b, 0, -1, idx[0]]
                        item_to_change[b, 0, -1, msg_n+offset] = word_n
                        offset += 1
            else:
                pass
            changed_input.append(item_to_change)     

        return changed_input

        
    def pgd_greedy_attack(self, inputs, attack,  **kwargs):
        """find the optimized attack by Greedy optimization
        """
        
        opt = self.config['pgd']
        steps = opt['steps']
        optm_lr = opt['lr']
        attackable_mask_layer = attack['attackable_mask_layer'].to(self.device)
                  
        assert self.syn_layer != None
        
        orig_text = inputs[0].detach().clone().long()
        inputs[0] = self.embedding(orig_text)     # trasnform text id to 
        embed_mask = attackable_mask_layer(orig_text)   # If word is in Glove
        
        # find syns
        syn_id, syn_mask = self.syn_layer(orig_text)
        syn_embd = self.embedding(syn_id)
        
        if opt['attack_type'] == 'deletion':    # change id to pad, and mask to 1
            syn_id, syn_mask = torch.ones(syn_id.shape, device=self.device), torch.ones(syn_mask.shape, device=self.device)
        
        if len(syn_embd.shape) - len(syn_mask.shape) == 1:
            syn_mask.unsqueeze_(-1)
        
        batch_size, nticker, nday, nmsg, nword, embed_dim = inputs[0].shape
        batch_size, nticker, nday, nmsg, nword, num_syn, embed_dim = syn_embd.shape

        # initialize attack variable
        mw = torch.ones(batch_size, nticker, nday, nmsg, 1, 1, device=self.device)
        zw = torch.ones(batch_size, nticker, nday, nmsg, nword, 1, device=self.device)
        rw = torch.ones(batch_size, nticker, nday, nmsg, nword, num_syn, 1,  device=self.device)
       
        # nn.init.kaiming_normal_(mw, mode='fan_out'), nn.init.kaiming_normal_(zw,  mode='fan_out'), nn.init.kaiming_normal_(rw,  mode='fan_out')
        nn.init.kaiming_normal_(mw), nn.init.kaiming_normal_(zw), nn.init.kaiming_normal_(rw)
        # nn.init.uniform_(mw), nn.init.uniform_(zw), nn.init.uniform_(rw)
        
        if opt['attack_type'] == 'replacement':
            rsoft = True
        elif opt['attack_type'] == 'deletion':
            rw = torch.zeros(batch_size, nticker, nday, nmsg, nword, num_syn, 1,  device=self.device)
            rw[:, :, :, :, :, 0] = 1
            rsoft = False
        
        rw = Variable(rw, requires_grad=True)
                
        # initialize optimizer         
        r_optimizer = init_optimizer([rw], opt['optim'], lr=optm_lr * 10, weight_decay=opt['weight_decay'])
        creterion = nn.NLLLoss(reduce=False)
        
        if opt['schedule_step'] != 0:
            self.r_scheduler = StepLR(r_optimizer, step_size=opt['schedule_step'], gamma=opt['schedule_gamma'])
        
        orig_pred_ = self._major_vote(inputs)[0]
        orig_label = orig_pred_[:,-1,:].argmax(-1)
        true_label = inputs[-1][:, 0, -1]
        
        orig_correct = orig_label == true_label
        if torch.sum(orig_correct) == 0:
            return batch_size, 0, 0, {}
        
        first_corr = 0
        for idx in range(batch_size):
            if orig_correct[idx] == 1:
                break
            else:
                first_corr += 1
                                                    
        loss_smooth = opt['smooth']
        alternative = opt['alt']
        sparsity = opt['sparsity']
        projection = opt['projection']
        fix = opt['fix']
        
        loss_file = {}
                            
        mnorm, znorm, rnorm = 0.0001, 0.0001, 0.0001
        
        mw_ = torch.ones(batch_size, nticker, nday, nmsg, 1, 1, device=self.device)
        zw_ = torch.ones(batch_size, nticker, nday, nmsg, nword, 1, device=self.device)
        
        m_num, z_num, r_num = 5, 5, opt['r_num']
        
        # project into probability space
        tweet_mask = inputs[2].view(batch_size, nticker, nday, nmsg, 1, 1)
        token_mask = inputs[4].view(batch_size, nticker, nday, nmsg, nword, 1)
        m_dis = topk_scatter(F.softmax(mw * tweet_mask + 500 * (tweet_mask - 1), dim=-3), m_num, -3, self.device)
        z_dis = topk_scatter(F.softmax(zw * token_mask + 500 * (token_mask * embed_mask - 1), dim=-2)    , z_num, -2, self.device)
        
        top_tweets, top_tokens = torch.zeros(mw_.shape), torch.zeros(zw_.shape)
        for step in range(steps):
            
            if opt['attack_type'] == 'replacement':            
                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], 
                                    mw_, zw_, rw, syn_embd, syn_mask, embed_mask, 
                                    msoft=False, zsoft=False, rsoft=True, concat=opt['concat'])
                pred_ = self.normal_forward(adv_inputs)[0]
                atk_loss = - creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                        
                loss_file[step] = atk_loss.cpu().detach().numpy().tolist()
                atk_loss *= orig_correct
            
                # sparsity
                sparsity_loss = sparsity * self._calc_sparse_loss(inputs, mw, zw, rw, syn_mask, opt['atk_days'], msoft=False, zsoft=False, rsoft=rsoft)
                sparsity_loss *= orig_correct
            
                atk_loss = atk_loss + sparsity_loss    # only correct prediction will be attacked
                
                # u optimization
                r_optimizer.zero_grad()
                atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
                r_optimizer.step()
                
            # m optimization by find the largest loss
            tweet_counts = inputs[1].detach().clone().cpu().numpy()
            token_counts = inputs[3].detach().clone().cpu().numpy()
            for i in range(batch_size):
                                
                if not orig_correct[i]:
                    continue
                
                tweet_count = int(tweet_counts[i, 0, -1])
                
                # find best tweet by iterative search
                if tweet_count > 1:
                    mw_mask = torch.zeros(tweet_count, nticker, nday, nmsg, 1, 1, device=self.device)
                    mw_mask.data[:, 0, -1, :tweet_count, 0, 0] = torch.eye(tweet_count)
                    inputs_temp = []
                    
                    for p in inputs:
                        inputs_temp.append(p[i].detach().clone().repeat([tweet_count]+[1]*(p.dim()-1)))
                    adv_inputs = self._ensemble_adv_tweets(inputs_temp, opt['atk_days'], mw_mask, 
                                                z_dis[[i]].repeat([tweet_count]+[1]*(z_dis.dim()-1)), 
                                                rw[[i]].repeat([tweet_count]+[1]*(rw.dim()-1)), 
                                                syn_embd[[i]].repeat([tweet_count]+[1]*(syn_embd.dim()-1)), 
                                                syn_mask[[i]].repeat([tweet_count]+[1]*(syn_mask.dim()-1)), 
                                                embed_mask[[i]].repeat([tweet_count]+[1]*(embed_mask.dim()-1)), 
                                                msoft=False, zsoft=False, rsoft=True, concat=opt['concat'])
                
                    pred_ = self.normal_forward(adv_inputs)[0]
                    loss = creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                    
                    m_idx = torch.topk(loss, dim=0, k=min(m_num, tweet_count))[1]
                    m_dis.data[i,:,:,:,:,:] = 0.0
                    m_dis[i, 0, opt['atk_days'], m_idx, 0, 0] = 1.0
                    
                else:
                    m_dis.data[i,:,:,:,:,:] = 0.0
                    m_dis[i, 0, opt['atk_days'], 0, 0, 0] = 1.0
                    m_idx = [0]
                
                # collect the ranking of select tweets
                for k in range(len(m_idx)):
                    top_tweets[i, 0, -1, m_idx[k], 0, 0] = len(m_idx) - k
                    
                # find the best token by search
                for j in range(tweet_count):
                                        
                    if m_dis[i, 0, opt['atk_days'], j, 0, 0] == 0:
                        continue
                             
                    token_amount = int(token_counts[i, 0, opt['atk_days'], j])
                    
                    if token_amount > 1:
                        zw_mask = z_dis[[i]].detach().clone().repeat([token_amount]+[1]*(z_dis.dim()-1))
                        zw_mask.data[:, 0, -1, j, :token_amount, 0] = torch.eye(token_amount)
                        inputs_temp = []
                        for p in inputs:
                            inputs_temp.append(p[[i]].detach().clone().repeat([token_amount]+[1]*(p.dim()-1)))
                        adv_inputs = self._ensemble_adv_tweets(inputs_temp, opt['atk_days'], 
                            m_dis[[i]].repeat([token_amount]+[1]*(m_dis.dim()-1)), 
                            zw_mask, 
                            rw[[i]].repeat([token_amount]+[1]*(rw.dim()-1)), 
                            syn_embd[[i]].repeat([token_amount]+[1]*(syn_embd.dim()-1)), 
                            syn_mask[[i]].repeat([token_amount]+[1]*(syn_mask.dim()-1)), 
                            embed_mask[[i]].repeat([token_amount]+[1]*(embed_mask.dim()-1)),
                            msoft=False, zsoft=False, rsoft=True, concat=opt['concat'])
                        
                        pred_ = self.normal_forward(adv_inputs)[0]
                        loss = creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                        loss = loss - 500 * (1 - embed_mask[i, 0, opt['atk_days'], j, :token_amount, 0])
                        z_idx = torch.topk(loss.squeeze_(), dim=0, k=min(z_num, token_amount))[1]
                        z_dis.data[i,:,:,j,:,:] = 0.0
                        z_dis.data[i,:,opt['atk_days'],j,z_idx,:] = 1.0
                    else:
                        z_idx = [0]
                        z_dis.data[i,:,:,j,:,:] = 0.0
                        z_dis.data[i,:,opt['atk_days'],j,0,:] = 1.0
                    
                    for k in range(len(z_idx)):
                        top_tokens[i, 0, -1, j, z_idx[k], 0] = len(z_idx) - k
                        
                assert m_dis[i, 0, opt['atk_days'], :, 0, 0].sum() == min(m_num, tweet_count)
                if i == first_corr:
                    print('step {} | (m, z): {}'.format(step, 
                            (m_idx, z_idx)))
        
        m_num, z_num, r_num = opt['m_num'], opt['z_num'], opt['r_num']   
            
        # project into probability space
        rw = F.softmax(rw * syn_mask + 500 * (syn_mask - 1), dim=-2)
        rw = topk_scatter(rw, r_num, -2, self.device)
        
        with open('./log/attack/{}_{}_concat{}_greedy_opt_{}_epoch{}_fix{}_type_{}_spa{}_alt{}_sm{}_m{}_z{}_r{}_step{}_lr{}_schedule{}_{}_proj{}/loss_log.json'.format(
                opt['prefix'], opt['attack_corpus'], opt['concat'], opt['model_name'], opt['ckpt_epoch'], opt['fix'], opt['attack_type'], opt['sparsity'], opt['alt'], opt['smooth'], opt['m_num'], 
                opt['z_num'], opt['r_num'], opt['steps'], opt['lr'], opt['schedule_step'], opt['schedule_gamma'], opt['projection']), 'a') as f:
            f.write(json.dumps(loss_file))
            f.write('\n')
        
        # print('m/r norm: {:.3f} | m/z norm: {:.3f}'.format(mnorm/rnorm, mnorm/znorm))
        attack_res = {}
        for m in range(0, m_num+1):
            for z in range(0, z_num+1):
                
                attack_res[(m,z)] = {}
                
                m_dis = topk_scatter(top_tweets, m, -3, self.device)
                z_dis = topk_scatter(top_tokens, z, -2, self.device)       
                r_dis = topk_scatter(rw, 1, -2, self.device) 
                adv_inputs = self._ensemble_adv_tweets(
                    inputs, opt['atk_days'], m_dis, z_dis, r_dis, syn_embd, syn_mask, embed_mask, 
                    msoft=False, zsoft=False, rsoft=False, concat=opt['concat'])
                
                adv_pred_ = self._major_vote(adv_inputs)[0]
                adv_label = adv_pred_[:, -1, :].argmax(-1)
                adv_wrong = adv_label != true_label
                
                attack_res[(m,z)]['attack_log'] = self.pgd_attack_record(m_dis, z_dis, r_dis, orig_text, inputs, syn_id, orig_label, true_label, adv_label)
                attack_res[(m,z)]['succ_num'] = torch.sum(adv_wrong * orig_correct).cpu().item()
                attack_res[(m,z)]['atk_num'] = torch.sum(orig_correct).cpu().item()

        return batch_size, attack_res
        
        
    def pgd_joint_attack(self, inputs, attack, **kwargs):
        """Joint find optimized attack for current inputs.
        Args:
            inputs ([type]): [description]
        """
        
        opt = self.config['pgd']
        steps = opt['steps']
        optm_lr = opt['lr']
        attackable_mask_layer = attack['attackable_mask_layer'].to(self.device)
        fin_mask_layer = attack['fin_mask_layer'].to(self.device)
        
        assert self.syn_layer != None
        
        orig_text = inputs[0].detach().clone().long()
        inputs[0] = self.embedding(orig_text)     # trasnform text id to 
        embed_mask = attackable_mask_layer(orig_text)   # If word is in Glove
        fin_mask = fin_mask_layer(orig_text)
        
        # embed_mask = fin_mask * embed_mask
        
        # find syns
        syn_id, syn_mask = self.syn_layer(orig_text)
        syn_embd = self.embedding(syn_id)
        
        if opt['attack_type'] == 'deletion':    # change id to pad, and mask to 1
            syn_id, syn_mask = torch.ones(syn_id.shape, device=self.device), torch.ones(syn_mask.shape, device=self.device)
        
        if len(syn_embd.shape) - len(syn_mask.shape) == 1:
            syn_mask.unsqueeze_(-1)
        
        batch_size, nticker, nday, nmsg, nword, embed_dim = inputs[0].shape
        batch_size, nticker, nday, nmsg, nword, num_syn, embed_dim = syn_embd.shape

        # initialize attack variable
        mw = torch.ones(batch_size, nticker, nday, nmsg, 1, 1, device=self.device)
        zw = torch.ones(batch_size, nticker, nday, nmsg, nword, 1, device=self.device)
        rw = torch.ones(batch_size, nticker, nday, nmsg, nword, num_syn, 1,  device=self.device)
       
        # nn.init.kaiming_normal_(mw, mode='fan_out'), nn.init.kaiming_normal_(zw,  mode='fan_out'), nn.init.kaiming_normal_(rw,  mode='fan_out')
        nn.init.kaiming_normal_(mw), nn.init.kaiming_normal_(zw), nn.init.kaiming_normal_(rw)
        # nn.init.uniform_(mw), nn.init.uniform_(zw), nn.init.uniform_(rw)
        
        # zw = zw + torch.abs(zw) * (1 + fin_mask)
        
        if opt['attack_type'] == 'replacement':
            rsoft = True
        elif opt['attack_type'] == 'deletion':
            rw = torch.zeros(batch_size, nticker, nday, nmsg, nword, num_syn, 1,  device=self.device)
            rw[:, :, :, :, :, 0] = 1
            rsoft = False
        else:
            raise Exception('Found unknown attack type!')
        
        mw = Variable(mw, requires_grad=True)
        zw = Variable(zw, requires_grad=True)
        rw = Variable(rw, requires_grad=True)
                
        # initialize optimizer         
        m_optimizer = init_optimizer([mw], opt['optim'], lr=optm_lr, weight_decay=opt['weight_decay'])
        z_optimizer = init_optimizer([zw], opt['optim'], lr=optm_lr * 5, weight_decay=opt['weight_decay'])
        r_optimizer = init_optimizer([rw], opt['optim'], lr=optm_lr * 10, weight_decay=opt['weight_decay'])
        creterion = nn.NLLLoss(reduce=False)
        
        if opt['schedule_step'] != 0:
            self.r_scheduler = StepLR(r_optimizer, step_size=opt['schedule_step'], gamma=opt['schedule_gamma'])
            self.m_scheduler = StepLR(m_optimizer, step_size=opt['schedule_step'], gamma=opt['schedule_gamma'])
            self.z_scheduler = StepLR(z_optimizer, step_size=opt['schedule_step'], gamma=opt['schedule_gamma'])
        
        # orig_pred_ = self.normal_forward(inputs)[0]
        orig_pred_ = self._major_vote(inputs)[0]
        orig_label = orig_pred_[:,-1,:].argmax(-1)
        true_label = inputs[-1][:, 0, -1]
        
        orig_correct = orig_label == true_label
        if torch.sum(orig_correct) == 0:
            return batch_size, 0, 0, {}
        
        first_corr = 0
        for idx in range(batch_size):
            if orig_correct[idx] == 1:
                break
            else:
                first_corr += 1
                                                    
        loss_smooth = opt['smooth']
        alternative = opt['alt']
        sparsity = opt['sparsity']
        projection = opt['projection']
        fix = opt['fix']
        
        loss_file = {}
        
        opt_concat = opt['concat']
                    
        if fix:
            msoft = False
            zw.data[:,:,:,:,:,:] = 0.0
            zw.data[:,:,opt['atk_days'], 0, 0,:] = 1.0
            mw.data[:,:,:,:,:,:] = 0.0
            mw.data[:, 0, opt['atk_days'], 0, 0, 0] = 1.0
        else:
            msoft = True
        
        mnorm, znorm, rnorm = 0.0001, 0.0001, 0.0001
        
        for step in range(steps):
            # ensemble new inputs
            
            alt_schedule = ['mw'] * 1 + ['zw'] * 1 + ['rw'] * 1
            mw_ = torch.ones(batch_size, nticker, nday, nmsg, 1, 1, device=self.device)
            zw_ = torch.ones(batch_size, nticker, nday, nmsg, nword, 1, device=self.device)
            
            if alternative:
                                
                while alt_schedule:
                    
                    if alt_schedule[0] == 'mw':
                        adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw, zw, rw, syn_embd, syn_mask, embed_mask,
                                                               msoft=msoft, zsoft=True, rsoft=rsoft, concat=opt_concat)
                        pred_ = self.normal_forward(adv_inputs)[0]
                        atk_loss = - creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                        
                        if loss_smooth:
                            for s in range(10):
                                # generate uniform sample
                                conv_mw = mw + 0.01 * torch.rand(mw.shape, device=self.device)
                                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], conv_mw, zw, rw, syn_embd, syn_mask, 
                                                                       msoft=msoft, zsoft=True, rsoft=rsoft, concat=opt_concat)
                                pred_ = self.normal_forward(adv_inputs)[0]
                                atk_loss += (- creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long()))
                            atk_loss = atk_loss / 11
                        
                        atk_loss *= orig_correct
                    
                        # sparsity
                        sparsity_loss = sparsity * self._calc_sparse_loss(inputs, mw, zw, rw, syn_mask, opt['atk_days'], msoft=msoft, zsoft=True, rsoft=rsoft)
                        sparsity_loss *= orig_correct
                    
                        atk_loss = atk_loss + sparsity_loss    # only correct prediction will be attacked
                        
                        m_optimizer.zero_grad()
                        atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
                        m_optimizer.step()

                    elif alt_schedule[0] == 'zw':
                        adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw_, zw, rw, 
                                                               syn_embd, syn_mask, msoft=False, zsoft=True, rsoft=rsoft, concat=opt_concat)
                        pred_ = self.normal_forward(adv_inputs)[0]
                        atk_loss = - creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                        
                        if loss_smooth:
                            for s in range(10):
                                # generate uniform sample
                                conv_zw = zw + 0.01 * torch.rand(zw.shape, device=self.device)
                                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw_, conv_zw, rw, 
                                                                       syn_embd, syn_mask, msoft=False, zsoft=True, rsoft=rsoft, concat=opt_concat)
                                pred_ = self.normal_forward(adv_inputs)[0]
                                atk_loss += (- creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long()))
                            atk_loss = atk_loss / 11
                        
                        atk_loss *= orig_correct
                    
                        # sparsity
                        sparsity_loss = sparsity * self._calc_sparse_loss(inputs, mw, zw, rw, syn_mask, opt['atk_days'], msoft=False, zsoft=True, rsoft=rsoft)
                        sparsity_loss *= orig_correct
                    
                        atk_loss = atk_loss + sparsity_loss    # only correct prediction will be attacked
                        
                        z_optimizer.zero_grad()
                        atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
                        z_optimizer.step()
                        # print('z step:', zw.grad[orig_correct, 0, -1, 0, :5])
                        
                    if alt_schedule[0] == 'rw':
                        adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw_, zw_, rw, 
                                                               syn_embd, syn_mask, msoft=False, zsoft=False, rsoft=rsoft, concat=opt_concat)
                        pred_ = self.normal_forward(adv_inputs)[0]
                        atk_loss = - creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                        
                        if loss_smooth:
                            for s in range(10):
                                # generate uniform sample
                                conv_rw = rw + 0.01 * torch.rand(rw.shape, device=self.device)
                                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw_, zw_, conv_rw, 
                                                                       syn_embd, syn_mask, msoft=False, zsoft=False, rsoft=rsoft, concat=opt_concat)
                                pred_ = self.normal_forward(adv_inputs)[0]
                                atk_loss += (- creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long()))
                            atk_loss = atk_loss / 11
                        
                        atk_loss *= orig_correct
                    
                        # sparsity
                        sparsity_loss = sparsity * self._calc_sparse_loss(inputs, mw, zw, rw, syn_mask, opt['atk_days'], msoft=False, zsoft=False, rsoft=rsoft)
                        sparsity_loss *= orig_correct
                    
                        atk_loss = atk_loss + sparsity_loss    # only correct prediction will be attacked
                        
                        r_optimizer.zero_grad()
                        atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
                        r_optimizer.step()
                        # print('r step:', rw.grad[orig_correct, 0, -1, 0, :5])
                        
                    alt_schedule.pop(0)
                
                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw, zw, rw, syn_embd, 
                                                       syn_mask, msoft=msoft, zsoft=msoft, rsoft=rsoft, concat=opt_concat)
                pred_ = self.normal_forward(adv_inputs)[0]
                atk_loss = - creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                
                loss_file[step] = atk_loss.cpu().detach().numpy().tolist()
                
            else:
                                                    
                # project back to probability space
                
                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw, zw, rw, syn_embd, syn_mask, embed_mask, 
                                                       msoft=msoft, zsoft=msoft, rsoft=rsoft, concat=opt_concat)
                pred_ = self.normal_forward(adv_inputs)[0]
                atk_loss = - creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long())
                
                if loss_smooth:
                    for s in range(10):
                        # generate uniform sample
                        conv_mw = mw + 0.01 * torch.rand(mw.shape, device=self.device)
                        conv_zw = zw + 0.01 * torch.rand(zw.shape, device=self.device)
                        conv_rw = rw + 0.01 * torch.rand(rw.shape, device=self.device)
                        adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], conv_mw, conv_zw, conv_rw, 
                                                syn_embd, syn_mask, embed_mask, msoft=False, rsoft=True, concat=opt_concat)
                        pred_ = self.normal_forward(adv_inputs)[0]
                        atk_loss += (- creterion(pred_[:, -1, :], adv_inputs[-1].squeeze(1)[:, -1].long()))
                    atk_loss = atk_loss / 11
                
                # m_optimizer.zero_grad()
                # z_optimizer.zero_grad()
                # r_optimizer.zero_grad()
                
                atk_loss *= orig_correct * 10
                loss_file[step] = atk_loss.cpu().detach().numpy().tolist()
                
                # atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
                
                # sparsity
                sparsity_loss = sparsity * self._calc_sparse_loss(inputs, mw, zw, rw, syn_mask, opt['atk_days'], msoft=msoft, zsoft=msoft, rsoft=rsoft)
                sparsity_loss *= orig_correct
                
                atk_loss = atk_loss + sparsity_loss    # only correct prediction will be attacked
                
                # print('NLE || m grad: {:.6f} | m: {:.6f} | z grad: {:6f} | z: {:.6f} | r grad: {:.6f} | r: {:.4f} | seleted: {} | loss: {:.4f} | spa: {:.4f}'.format(
                #         custom_norm(mw.grad[orig_correct,:,-1], torch.sum(inputs[2][orig_correct, 0, -1])).item(), 
                #         custom_norm(mw[orig_correct,:,-1], torch.sum(inputs[2][orig_correct, 0, -1])).item(), 
                #         custom_norm(zw.grad[orig_correct,:,-1], torch.sum(inputs[4][orig_correct, 0, -1])).item(), 
                #         custom_norm(zw[orig_correct,:,-1], torch.sum(inputs[4][orig_correct, 0, -1])).item(), 
                #         custom_norm(rw.grad[orig_correct,:,-1,: ,:], torch.sum(inputs[4][orig_correct, 0, -1]*15)).item(), 
                #         custom_norm(rw[orig_correct,:,-1, :,:], torch.sum(inputs[4][orig_correct, 0, -1]*15)).item(), 
                #         (0, 0, 0),
                #         np.mean(loss_file[step]), 
                #         sparsity_loss.mean()
                #         ))
                
                m_optimizer.zero_grad()
                z_optimizer.zero_grad()
                r_optimizer.zero_grad()
                
                # sparsity_loss.backward(retain_graph=True, gradient=torch.ones_like(sparsity_loss))
                
                # print('SPA || m grad: {:.6f} | m: {:.6f} | z grad: {:6f} | z: {:.6f} | r grad: {:.6f} | r: {:.4f} | seleted: {} | loss: {:.4f} | spa: {:.4f}'.format(
                #         custom_norm(mw.grad[orig_correct,:,-1], torch.sum(inputs[2][orig_correct, 0, -1])).item(), 
                #         custom_norm(mw[orig_correct,:,-1], torch.sum(inputs[2][orig_correct, 0, -1])).item(), 
                #         custom_norm(zw.grad[orig_correct,:,-1], torch.sum(inputs[4][orig_correct, 0, -1])).item(), 
                #         custom_norm(zw[orig_correct,:,-1], torch.sum(inputs[4][orig_correct, 0, -1])).item(), 
                #         custom_norm(rw.grad[orig_correct,:,-1,: ,:], torch.sum(inputs[4][orig_correct, 0, -1]*15)).item(), 
                #         custom_norm(rw[orig_correct,:,-1, :,:], torch.sum(inputs[4][orig_correct, 0, -1]*15)).item(), 
                #         (0, 0, 0),
                #         np.mean(loss_file[step]), 
                #         sparsity_loss.mean()
                #         ))
                
                atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
                
                if opt['schedule_step'] != 0:
                    self.r_scheduler.step() 
                    self.m_scheduler.step() 
                    self.z_scheduler.step() 
                                    
                # atk_loss.backward(retain_graph=True, gradient=torch.ones_like(atk_loss))
    
                if fix == 0:
                    m_optimizer.step()
                    z_optimizer.step()
                
                if opt['attack_type'] == 'replacement':
                    r_optimizer.step()
 
                if mw.grad != None:
                    mnorm += custom_norm(mw.grad[orig_correct])
                if zw.grad != None:
                    znorm += custom_norm(zw.grad[orig_correct])
                if rw.grad != None:
                    rnorm += custom_norm(rw.grad[orig_correct])
                
                if projection == 1:     # clip into [0,1]
                    rw.data = torch.tensor(np.clip(rw.cpu().detach().numpy(), 0, 1)).data 
                    # rw.data = topk_scatter(rw, 1, -2, self.device).data   # back to discrete space        
                elif projection == 2:   # bisection
                    fcn = lambda mu, a=rw.cpu().detach().numpy(): np.sum(np.max())
                elif projection == 3:   # to be determined
                    pass
                                    
            m_idx = torch.argmax(mw[first_corr, 0, -1, :, 0, 0]).long().cpu().item()
            z_idx = torch.argmax(zw[first_corr, 0, -1, m_idx, :, 0]).long().cpu().item()
            r_idx = torch.argmax(rw[first_corr, 0, -1, m_idx, z_idx, :, 0]).long().cpu().item()      
            # print('m grad: {:.6f} | m: {:.6f} | z grad: {:6f} | z: {:.6f} | r grad: {:.6f} | r: {:.4f} | seleted: {} | loss: {:.4f} | spa: {:.4f}'.format(
            #             custom_norm(mw.grad[orig_correct,:,-1], torch.sum(inputs[2][orig_correct, 0, -1])).item(), 
            #             custom_norm(mw[orig_correct,:,-1], torch.sum(inputs[2][orig_correct, 0, -1])).item(), 
            #             custom_norm(zw.grad[orig_correct,:,-1], torch.sum(inputs[4][orig_correct, 0, -1])).item(), 
            #             custom_norm(zw[orig_correct,:,-1], torch.sum(inputs[4][orig_correct, 0, -1])).item(), 
            #             custom_norm(rw.grad[orig_correct,:,-1,: ,:], torch.sum(inputs[4][orig_correct, 0, -1]*15)).item(), 
            #             custom_norm(rw[orig_correct,:,-1, :,:], torch.sum(inputs[4][orig_correct, 0, -1]*15)).item(), 
            #             (m_idx, z_idx, r_idx),
            #             np.mean(loss_file[step]), 
            #             sparsity_loss.mean()
            #             ))
            
            # TODO: look into different gradient approach
        
        # generate and evaluate attack results    
        m_num, z_num, r_num = opt['m_num'], opt['z_num'], opt['r_num']
        
        # project into probability space
        tweet_mask = inputs[2].view(batch_size, nticker, nday, nmsg, 1, 1)
        token_mask = inputs[4].view(batch_size, nticker, nday, nmsg, nword, 1) * fin_mask
        mw = F.softmax(mw * tweet_mask + 500 * (tweet_mask - 1), dim=-3)
        zw = F.softmax(zw * token_mask * embed_mask + 500 * (token_mask * embed_mask - 1), dim=-2)    
        rw = F.softmax(rw * syn_mask + 500 * (syn_mask - 1), dim=-2)
                
        # mw = topk_scatter(mw, m_num, -3, self.device)
        # zw = topk_scatter(zw, z_num, -2, self.device)
        # rw = topk_scatter(rw, r_num, -2, self.device)
    
        with open('./log/attack/{}_{}_concat{}_joint_opt_{}_epoch{}_fix{}_type_{}_spa{}_alt{}_sm{}_m{}_z{}_r{}_step{}_lr{}_schedule{}_{}_proj{}/loss_log.json'.format(
                opt['prefix'], opt['attack_corpus'], opt['concat'], opt['model_name'], opt['ckpt_epoch'], opt['fix'], opt['attack_type'], opt['sparsity'], opt['alt'], opt['smooth'], opt['m_num'], 
                opt['z_num'], opt['r_num'], opt['steps'], opt['lr'], opt['schedule_step'], opt['schedule_gamma'], opt['projection']), 'a') as f:
            f.write(json.dumps(loss_file))
            f.write('\n')
        
        # print('m/r norm: {:.3f} | m/z norm: {:.3f}'.format(mnorm/rnorm, mnorm/znorm))
        attack_res = {}
        for m in range(0, m_num+1):
            for z in range(0, z_num+1):
                
                attack_res[(m,z)] = {}

                mw_ = topk_scatter(mw, m, -3, self.device)
                # zw_ = topk_scatter(zw * (1+fin_mask), z, -2, self.device)
                zw_ = topk_scatter(zw, z, -2, self.device)
                rw_ = topk_scatter(rw, 1, -2, self.device)   
                adv_inputs = self._ensemble_adv_tweets(inputs, opt['atk_days'], mw_, zw_, rw_, syn_embd, syn_mask, embed_mask, 
                                                       msoft=False, zsoft=False, rsoft=False, concat=opt['concat'])
                
                adv_pred_ = self._major_vote(adv_inputs)[0]
                adv_label = adv_pred_[:, -1, :].argmax(-1)
                adv_wrong = adv_label != true_label
                
                attack_res[(m,z)]['attack_log'] = self.pgd_attack_record(mw_, zw_, rw_, orig_text, inputs, syn_id, orig_label, true_label, adv_label)
                attack_res[(m,z)]['succ_num'] = torch.sum(adv_wrong * orig_correct).cpu().item()
                attack_res[(m,z)]['atk_num'] = torch.sum(orig_correct).cpu().item()

        return batch_size, attack_res
    
    def _calc_sparse_loss(self, inputs, mw, zw, rw, syn_mask, atk_days, msoft=True, zsoft=True, rsoft=True):
        """calculate sparsity entropy loss
        Args:
            mw ([type]): [description]
            zw ([type]): [description]
            rw ([type]): [description]
        """
        
        batch_size, nticker, nday, nmsg, nword, embed_dim = inputs[0].shape
        tweet_mask = inputs[2].view(batch_size, nticker, nday, nmsg, 1, 1)[:,:,atk_days]
        token_mask = inputs[4].view(batch_size, nticker, nday, nmsg, nword, 1)[:,:,atk_days]
    
        tweet_p = F.softmax(mw[:, :, atk_days] * tweet_mask + 500 * (tweet_mask - 1), dim=-3)
        token_p = F.softmax(zw[:, :, atk_days] * token_mask + 500 * (token_mask - 1), dim=-2)
        syn_p = F.softmax(rw * syn_mask + 500 * (syn_mask - 1), dim=-2)[:,:,atk_days]
        
        m_loss, z_loss, r_loss = 0, 0, 0
        if msoft:
            m_loss = (tweet_mask*tweet_p*torch.sum(torch.log(tweet_p+0.0001)*tweet_mask, dim=-3, keepdim=True)).mean(-3).squeeze()
        if zsoft:
            z_loss = (tweet_mask*token_mask*token_p * torch.sum(torch.log(token_p+0.0001)*token_mask, dim=-2, keepdim=True)).mean((-2,-3)).squeeze()
        if rsoft:
            r_loss = ((tweet_mask*token_mask*(syn_p * torch.sum(torch.log(syn_p+0.0001)*syn_mask[:,:,atk_days], dim=-2, keepdim=True)).squeeze(-1)).mean((-1,-2,-3))).squeeze()   
        
        return -(m_loss + z_loss + r_loss)
        
    def _ensemble_adv_tweets(self, inputs, atk_days, mw, zw, rw, syn, syn_mask, avoid_mask, msoft=False, zsoft=False, rsoft=False, concat=True):
        """[summary]

        Args:
            orig_embd ([type]): [description]
            atk_days ([type]): [description]
            mw ([type]): [description]
            zw ([type]): [description]
            rw ([type]): [description]
            syn ([type]): [description]
            syn_mask ([type]): [description]
            avoid_mask ([int]): [mask for token to avoid, marked as 0]
        """
        batch_size, nticker, nday, nmsg, nword, embed_dim = inputs[0].shape
        
        syn_comb = self._calc_syn_convex_comb(rw, syn, syn_mask, dim=-2, soft=rsoft)

        tweet_mask = inputs[2].detach().clone().view(batch_size, nticker, nday, nmsg, 1, 1)
        token_mask = inputs[4].detach().clone().view(batch_size, nticker, nday, nmsg, nword, 1) * avoid_mask
        
        # ptbn_text = mw * zw * syn_comb + (1 - mw * zw) * (inputs[0].detach().clone())
        
        if msoft and zsoft:
            tweet_p = F.softmax(mw * tweet_mask + 500 * (tweet_mask - 1), dim=-3)
            token_p = F.softmax(zw * token_mask + 500 * (token_mask - 1), dim=-2) 
            ptbn_text = tweet_p * token_p * syn_comb + (1 -  tweet_p * token_p) * (inputs[0].detach().clone())
        elif msoft and not zsoft:     # z is hard  
            tweet_p = F.softmax(mw * tweet_mask + 500 * (tweet_mask - 1), dim=-3)                 
            ptbn_text = tweet_p * zw * syn_comb + (1 -  tweet_p * zw) * (inputs[0].detach().clone())   
        elif not msoft and zsoft:
            token_p = F.softmax(zw * token_mask + 500 * (token_mask - 1), dim=-2) 
            ptbn_text = mw * token_p * syn_comb + (1 -  mw * token_p) * (inputs[0].detach().clone())    
        else:
            ptbn_text = mw * zw * syn_comb + (1 - mw * zw) * (inputs[0].detach().clone())
        if concat:
            adv_text = torch.cat([inputs[0].detach().clone(), ptbn_text], dim=-3)    # concat tweet
            adv_twt_count, adv_twt_mask, adv_word_count, adv_word_mask = self._adv_input_associates(inputs[2].detach().clone(), inputs[3].detach().clone(), inputs[4].detach().clone(), mw, atk_days, msoft)
            return [adv_text, adv_twt_count, adv_twt_mask, adv_word_count, adv_word_mask, inputs[5], inputs[6]]
        else:
        # # use old text
        # # adv_twt_count, adv_twt_mask, adv_word_count, adv_word_mask = input[1], input[2], input[3], input[4]
            return [ptbn_text] + inputs[1:]

    def _calc_syn_convex_comb(self, rw, syn, syn_mask, dim=-2, soft=True):
        """Calculate convex combination
        Args:
            mw ([type]): [description]
            zw ([type]): [description]
            rw ([type]): [description]
            syn ([type]): [description]
            y ([type], optional): [description]. Defaults to None.
            dim (int, optional): [description]. Defaults to -1.
        """
        
        # return (rw * syn_mask * syn.detach()).sum(dim)  
        
        if soft:
            syn_p = F.softmax(rw * syn_mask + 500 * (syn_mask - 1), dim=dim)
        else:
            syn_p = syn_mask * rw
        syn_comb = (syn_p * syn.detach()).sum(dim)  
        
        return syn_comb
        
    def _adv_input_associates(self, orig_tweet_mask, orig_word_count, orig_word_mask, mw, atk_days, msoft=True):
        """calculate the input mask, message count, word count for adversarial input.
        Args:
            inputs (tuple): original input tuple
            m (tensor): tweet selection tensor, [batch_size, max_n_tweet]
            soft (bool): soft or hard encoding of m and z
        """
        
        ptbn_tweet_mask = torch.zeros(orig_tweet_mask.shape, device=self.device)
        ptbn_tweet_mask.data[:, :, atk_days, :] = 1
        if msoft:
            adv_tweet_mask = torch.cat([orig_tweet_mask.detach().clone(), ptbn_tweet_mask * orig_tweet_mask.detach().clone()], dim=3)
        else:
            adv_tweet_mask = torch.cat([orig_tweet_mask.detach().clone(), ptbn_tweet_mask * mw.view(orig_tweet_mask.shape)], dim=-1)
        
        adv_tweet_count = torch.sum(adv_tweet_mask, dim=-1)
        # replacement will not change word count
        
        ptbn_word_mask = torch.zeros(orig_word_mask.shape, device=self.device)
        ptbn_word_mask.data[:, :, atk_days, :, :] = 1
        adv_word_mask = torch.cat([orig_word_mask.detach().clone(), ptbn_word_mask * orig_word_mask.detach().clone()], dim=-2)   # concat at tweet level
        adv_word_count = torch.cat(2 * [orig_word_count.detach().clone()], dim=-1)       

        return [adv_tweet_count, adv_tweet_mask, adv_word_count, adv_word_mask]
    
    def pgd_attack_record(self, mw, zw, rw, orig_text, inputs, syn_id, orig_label, true_label, adv_label):
        """generate attack information dictionary"""
        
        orig_correct = orig_label == true_label
        
        attack_log = {}
        
        batch_size, nticker, nday, nmsg, nword, num_syn, _ = rw.shape 
        
        for i in range(batch_size):
            attack = {}
            attack['original_label'], attack['adv_label'] = orig_label[i].cpu().item(), adv_label[i].cpu().item()
            attack['true_label'] = true_label[i].cpu().item()
            
            if orig_correct[i] == True:
                attack['attack'] = True
                attack['tweet_amount_attack_day'] = int(inputs[1][i, 0, -1].int().cpu().item())
                attack['tweet_amount_total'] = int(torch.sum(inputs[1][i, 0, :]).int().cpu().item())
                attack['token_amount_attack_day'] = torch.sum(inputs[3][i, 0, -1, :attack['tweet_amount_attack_day']]).cpu().item()
                attack['token_amount_total'] = torch.sum(inputs[3][i, 0, :, :]).cpu().item()
                attack_replacement, original_tweets, attack_positions = [], [], []
                for m in range(attack['tweet_amount_attack_day']):
                    if mw[i, 0, -1, m, 0, 0] == 1:       # adversarial message
                        token_count = int(inputs[3][i, 0, -1, m])
                        
                        orig_tweet = self.embedding.index2word(orig_text[i, 0, -1, m, :token_count].cpu().numpy().tolist())
                        original_tweets.append(orig_tweet)
                        
                        replaces, position = [], []
                        for n in range(token_count):
                            if zw[i, 0, -1, m, n, 0] == 1:
                                for r in range(num_syn):
                                    if rw[i, 0, -1, m, n, r, 0] == 1:
                                        replace = [int(orig_text[i, 0, -1, m, n].cpu().item()), int(syn_id[i, 0, -1, m, n, r].cpu().item())]
                                        replaces.append(self.embedding.index2word(replace))
                                        position.append([n, r])
                        
                        attack_replacement.append(replaces)
                        attack_positions.append(position)
      
                attack['original'] = original_tweets
                attack['adversarial'] = attack_replacement
                attack['position'] = attack_positions
            else:
                attack['attack'] = False
            attack_log[i] = attack
        return attack_log
    
if __name__ == '__main__':
    
    bm = BaseModel(None, 'cpu', None)
    
    batch_size, nticker, nday, nmsg, nword, embed_dim, num_syn = [2, 1, 5, 30, 40, 50, 20]
    
    syn = torch.ones(batch_size, nticker, nday, nmsg, nword, num_syn, embed_dim)
    syn_mask = torch.ones(batch_size, nticker, nday, nmsg, nword, num_syn, 1)
    rw = torch.empty(batch_size, nticker, nday, nmsg, nword, num_syn, 1,  device=bm.device)         
    zw = torch.empty(batch_size, nticker, nday, nmsg, nword, 1, device=bm.device) 
    mw = torch.empty(batch_size, nticker, nday, nmsg, 1, 1,  device=bm.device)
    
    syn_comb = bm._calc_syn_convex_comb(rw, syn, syn_mask, dim=-2)
    
    a = mw * syn_comb
    
    tickers = ['MSFT']
    device = 'cpu'
    
    embedding = GloveEmbedding(resource=path.glove, n_d=config['model']['word_embed_size'], 
                                vocab_list=config['vocab'])
    embedding.to(device)
    
    dset = StockData(config['dates']['test_start_date'], config['dates']['test_end_date'], tickers, embedding, device)
    loader = DataLoader(dset, 2, shuffle=False)
    atk_days = [-1]
    
    for idx, sample in enumerate(loader, 0):
        if idx >= 1:
            break
        
        orig_text, orig_tweet_count, orig_tweet_mask, orig_word_count, orig_word_mask, technical, y = sample
        out = bm._ensemble_adv_tweets(sample, atk_days, mw, zw, rw, syn, syn_mask, soft=True)
        
        for o in out:
            print(o.shape)
            print(o[0][0][-1])