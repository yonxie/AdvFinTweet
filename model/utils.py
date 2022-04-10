# -*- coding: utf-8 -*-
"""
@file: utils
@author: Yong Xie
@time: 5/16/2021 2:29 PM
@Description: 
"""

import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.init as init
import io, os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textattack.shared import WordEmbedding
from nltk.corpus import wordnet, brown
from collections import Counter


class Synonymer(nn.Module):
    
    def __init__(self, embedding, syn_num=10, syn_file=None):
        super(Synonymer, self).__init__()
        self.syn_file = syn_file
        self.syn_mask_file = syn_file[:-4] + '_mask' + syn_file[-4:]
        self.syn_num = syn_num
        if (not os.path.exists(syn_file)) or (not os.path.exists(self.syn_mask_file)):
            self.embedding = embedding
            self._calc_syns_from_embd()
        else:
            self._read_syn_from_file()
        
    def _calc_syns_from_embd(self):
        """calculate synonums for the whole vocal""" 
        
        assert (not os.path.exists(self.syn_file)) or (not os.path.exists(self.syn_mask_file))
        
        print('initialize synonyms from scratch..')
                
        # calculate similarity matrix
        
        cutoff = [i for i in range(0, self.embedding.word_table.shape[0], 10000)] + [self.embedding.word_table.shape[0] + 1]
        
        self.syn_table = torch.zeros((self.embedding.word_table.shape[0], self.syn_num))
        for s, e in zip(cutoff[:-1], cutoff[1:]):
            sim_mat = np.dot(self.embedding.word_table[s:e, :], self.embedding.word_table.T)
            norm1 = np.linalg.norm(self.embedding.word_table[s:e,:], axis=1, keepdims=True)
            norm2 = np.linalg.norm(self.embedding.word_table, axis=1, keepdims=True)
            sim_mat = sim_mat / np.dot(norm1, norm2.T)
            self.syn_table[s:e, :] = torch.topk(torch.tensor(sim_mat), self.syn_num+1, dim=-1)[1][:, 1:]
        
        # Generate synonym layer    
        self.syn_layer = nn.Embedding(self.syn_table.shape[0], self.syn_num)
        self.syn_layer.weight.data.copy_(self.syn_table)
        
        # Generate synonyn mask (filter out unknown words and paddings)
        mask = torch.ones(self.syn_table.shape[0], self.syn_num)
        for i in range(self.syn_table.shape[0]):
            for j in range(self.syn_num):
                if int(self.syn_table[i,j]) not in self.embedding.embed_words:
                    mask[i, j] = 0
       
        self.syn_mask_layer = nn.Embedding(self.syn_table.shape[0], self.syn_num)
        self.syn_mask_layer.weight.data.copy_(mask)
        print('Number of instances with NO SYNONYMS: {}/{}; Average number of Syns: {}'.format(torch.sum(torch.sum(mask, 1)==0), self.syn_table.shape[0], torch.mean(torch.sum(mask, 1))))
        
        syn_files = open('fin_tweet_syn.txt', 'w')
        for i in range(self.syn_table.shape[0]):
            token = self.embedding.index2word([i])
            syns = self.embedding.index2word(self.syn_table[i].cpu().numpy().tolist())
            syn_files.write(str([token, syns]))
            syn_files.write('\n')
        syn_files.close()
        
        # store synonym table
        pd.DataFrame(self.syn_table.cpu().numpy()).to_csv(self.syn_file, index=None)
        pd.DataFrame(self.syn_mask_layer.weight.data.cpu().numpy()).to_csv(self.syn_mask_file, index=False)
        print('synonyms and mask are saved in {}'.format(self.syn_file))
        
    def _read_syn_from_file(self):
        """read synonyms from file"""

        print('initialize synosyms from saved files..')
        self.syn_table = torch.tensor(pd.read_csv(self.syn_file, header=None).values)
        syn_mask = torch.tensor(pd.read_csv(self.syn_mask_file, header=None).values)
        
        assert self.syn_table.shape[1] >= self.syn_num
        
        self.syn_layer = nn.Embedding(self.syn_table.shape[0], self.syn_num)
        self.syn_layer.weight.data.copy_(self.syn_table[:, :self.syn_num])
        self.syn_mask_layer = nn.Embedding(self.syn_table.shape[0], self.syn_num)
        self.syn_mask_layer.weight.data.copy_(syn_mask[:, :self.syn_num])
        
        print('Number of instances with NO SYNONYMS: {}/{}; Average number of Syns: {}'.format(torch.sum(torch.sum(syn_mask, 1)==0), self.syn_table.shape[0], torch.mean(torch.sum(syn_mask, 1))))
          
    def forward(self, idx):
        idx = torch.tensor(idx, device=self.syn_layer.weight.device)
        idx_shape, idx_device = list(idx.shape), idx.device
        idx_shape.append(self.syn_num)
        return  [self.syn_layer(idx).long(), self.syn_mask_layer(idx)]

def init_optimizer(params, optim, lr, weight_decay):
    """initialize optimizer.

    Args:
        params (list): list of parameters to optimaze
        optim (str): type of optimizer
        lr (float): learning rate
        weight_decay (float): weight_decay
    """
    if optim =="adam":
        optimizer=torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim =="sgd":
        optimizer=torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    else:
        raise Exception('found unexpected optimizer!')
    return optimizer

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        pass
        # init.normal_(m.weight.data, mean=1, std=0.02)
        # init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        try:
            init.normal_(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        print('skip embedding layer')
    else:
        pass


class GloveEmbedding(nn.Module, WordEmbedding):
    """A layer of a model that replaces word IDs with their embeddings.

    This is a useful abstraction for any nn.module which wants to take word IDs
    (a sequence of text) as input layer but actually manipulate words'
    embeddings.

    Requires some pre-trained embedding with associated word IDs.
    """

    def __init__(self, resource, n_d=50, vocab_list=None, normalize=False):
        super(GloveEmbedding, self).__init__()
        self.resource = resource
        self.n_d = n_d
        self.vocab = vocab_list
        self.init_word_table()    # generate 
        
        self.n_V, self.n_d = self.word_table.shape
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-1, 1)
        weight = self.embedding.weight
        weight.data[:self.n_V].copy_(torch.from_numpy(self.word_table))
        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))
        # Layer to generate mask for Glove word
        self.mask_layer = nn.Embedding(self.n_V, 1)
        self.mask_layer.weight.data.copy_(torch.from_numpy(self.glove_mask))
        
    def init_word_table(self):
        """
        # Initialize word table.
        """

        avoid_set = {'<number>', 'UNK', 'PAD', 'pad', '<ticker>', 'date'}

        self.word_table = np.random.rand(len(self.vocab)+4, self.n_d) * 2 - 1  # [-1.0, 1.0]
        self.glove_mask = np.zeros((len(self.vocab)+4, 1))
        vocab_id_dict = self._index_token(self.vocab)

        self.embed_words = {}
        with io.open(self.resource, 'r', encoding='utf-8') as f:
            for line in f:
                tuples = line.split()
                word, embed = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                if word in ['<unk>', 'unk']:  # unify UNK
                    word = 'UNK'
                if word in vocab_id_dict:
                    word_id = vocab_id_dict[word]
                    self.embed_words[word_id] = word
                    self.word_table[word_id] = torch.tensor(embed)
                    self.glove_mask[word_id, 0] = torch.tensor(1)
                    if word in avoid_set:
                        self.glove_mask[word_id, 0] = torch.tensor(0)
                    
        print('Glove embeddings: embed word: {}, total word: {}'.format(len(self.embed_words), self.word_table.shape[0]))
        self.vocab_id_dict = vocab_id_dict
        
        # change 
        
        # convert id to vocab
        self.id_vocab_dict = {}
        for k, v in self.vocab_id_dict.items():
            self.id_vocab_dict[v] = k            
        
    @staticmethod
    def _index_token(token_list):
        """
        # Generate index for the tokens used for the models.
        :return: dict with token as key and index as vakue
        """

        indexed_token_dict = dict()

        token_list_cp = list(token_list)  # un-change the original input
        
        token_list_cp.insert(0, 'UNK')    # for unknown tokens
        token_list_cp.insert(1, 'PAD')    # for unknown tokens
        
        for idx in range(len(token_list_cp)):
            indexed_token_dict[token_list_cp[idx]] = idx
        
        return indexed_token_dict
    
    def forward(self, input):
        return self.embedding(input)

    def mask(self, input):
        return self.mask_layer(input)
    
    def _convert_token(self, token):
        if token not in self.vocab_id_dict:
            token = 'UNK'
        return self.vocab_id_dict[token]
    
    def word2index(self, words):
        """ convert words to id
        Args:
            words ([list]): tokenized word list
        """
        if isinstance(words, str):
            return self._convert_token(words)
        else:
            return [self._convert_token(w) for w in words]
    
    def _convert_id(self, id):
        if id not in self.id_vocab_dict:
            id = 0
        return self.id_vocab_dict[id]    
    
    def index2word(self, ids):
        """convert ids to words
        
        Args:
            ids (list): list of ids
        """
        if torch.is_tensor(ids):
            ids = ids.data.cpu().numpy().ravel().tolist()
        return [self._convert_id(id) for id in ids]

class AttackMask(nn.Module, WordEmbedding):
    """A layer of binary embedding for valid attackable tokens."""

    def __init__(self, resource, vocab_list=None, avoid_set=set()):
        super(AttackMask, self).__init__()
        self.resource = resource
        self.vocab = vocab_list
        self.avoid_set = avoid_set
        self.init_mask_table()    # generate 
        
        self.n_V, self.n_d = self.attack_mask.shape
        
        # Layer to generate mask for Glove word
        self.embedding = nn.Embedding(self.n_V, 1)
        self.embedding.weight.data.copy_(torch.from_numpy(self.attack_mask))
        
        print('Total words: {};  Valid attack words: {}'.format(self.n_V, torch.sum(self.embedding.weight.data)))
        
    def init_mask_table(self):
        """
        # Initialize mask table.
        """

        self.attack_mask = np.zeros((len(self.vocab)+4, 1))
        vocab_id_dict = self._index_token(self.vocab)

        vocab_words = set()
        with io.open(self.resource, 'r', encoding='utf-8') as f:
            for line in f:
                tuples = line.split()
                word, _ = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                if word in ['<unk>', 'unk']:  # unify UNK
                    word = 'UNK'
                vocab_words.add(word)
                    
        self.vocab_id_dict = vocab_id_dict
        
        for w in self.vocab_id_dict:
            w_id = self.vocab_id_dict[w]
            if (w in vocab_words) and (w not in self.avoid_set) and (w[0] != '$'):
                self.attack_mask[w_id, 0] = 1

        # convert id to vocab
        self.id_vocab_dict = {}
        for k, v in self.vocab_id_dict.items():
            self.id_vocab_dict[v] = k            
        
    @staticmethod
    def _index_token(token_list):
        """
        # Generate index for the tokens used for the models.
        :return: dict with token as key and index as vakue
        """

        indexed_token_dict = dict()

        token_list_cp = list(token_list)  # un-change the original input
        
        token_list_cp.insert(0, 'UNK')    # for unknown tokens
        token_list_cp.insert(1, 'PAD')    # for unknown tokens
        
        for idx in range(len(token_list_cp)):
            indexed_token_dict[token_list_cp[idx]] = idx
        
        return indexed_token_dict
    
    def forward(self, input):
        return self.embedding(input)
    
    def _convert_token(self, token):
        if token not in self.vocab_id_dict:
            token = 'UNK'
        return self.vocab_id_dict[token]
    
    def word2index(self, words):
        """ convert words to id
        Args:
            words ([list]): tokenized word list
        """
        if isinstance(words, str):
            return self._convert_token(words)
        else:
            return [self._convert_token(w) for w in words]
    
    def _convert_id(self, id):
        if id not in self.id_vocab_dict:
            id = 0
        return self.id_vocab_dict[id]    
    
    def index2word(self, ids):
        """convert ids to words
        
        Args:
            ids (list): list of ids
        """
        if torch.is_tensor(ids):
            ids = ids.data.cpu().numpy().ravel().tolist()
        return [self._convert_id(id) for id in ids]

class WordWeight(nn.Module, WordEmbedding):
    """A layer of binary embedding for valid attackable tokens."""

    def __init__(self, resource, vocab_list=None, config={}, stopwords=set()):
        super(WordWeight, self).__init__()
        self.resource = resource
        self.vocab = vocab_list
        self.config = config
        self.stopwords = stopwords
        self.init_mask_table()    # generate 
        
        self.n_V, self.n_d = self.glove_mask.shape
        
        # Layer to generate mask for Glove word
        self.embedding = nn.Embedding(self.n_V, 1)
        self.embedding.weight.data.copy_(torch.from_numpy(self.glove_mask))
        
        print('{}: Total words: {};  Valid words: {}'.format(self.config['attack_corpus'], self.n_V, torch.sum(self.embedding.weight.data)))
        
    def init_mask_table(self):
        """
        # Initialize mask table.
        """
        brown_cate = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
                 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance','science_fiction']
        
        # read weight word
        if self.config['attack_corpus'] == 'fpb':
            with open('/shared/rsaas/yongxie2/glove/vocab_fin_phrasebank_frequency.json') as f:
                weight_vocab_dict = json.load(f)
        elif self.config['attack_corpus'] in brown_cate:
            weight_vocab_dict = Counter(brown.words(categories=self.config['attack_corpus']))
            def enrich_corpus_with_syn(corpus):
                """enrich corpus by syn"""
                new_corpus = {}
                for word in corpus:
                    new_corpus[word] = corpus[word]
                    for syn in wordnet.synsets(word):
                        for w in [l.name() for l in syn.lemmas()]:
                            if w not in new_corpus:
                                new_corpus[w] = corpus[word]    
                return new_corpus
            weight_vocab_dict = enrich_corpus_with_syn(weight_vocab_dict)
        elif self.config['attack_corpus'] == 'all':
            pass
        else:
            raise Exception('Found unknown attack corpus!', self.config['attack_corpus'])
        
        if self.config['attack_corpus'] != 'all':
        
            for token in self.stopwords:
                if token in weight_vocab_dict:
                    weight_vocab_dict.pop(token)
            
            # weight_freq = list(weight_vocab_dict.values())
            # cutoff = np.quantile(weight_freq, 0.8)
            weight_vocab = sorted(weight_vocab_dict.items(), key=lambda x: x[1], reverse=True)[:5000]
            
            self.word_weight = {}
            for w, freq in weight_vocab:
                self.word_weight[w] = freq
            
            print('attack set number in {} : {}'.format(self.config['attack_corpus'], len(self.word_weight)))
            # print('total num in {}: {}', len(weight_vocab_dict))

        # with io.open(self.resource, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         tuples = line.split()
        #         word, _ = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
        #         if word in ['<unk>', 'unk']:  # unify UNK
        #             word = 'UNK'
        #         if word in vocab_id_dict:
        #             word_id = vocab_id_dict[word]
        #             self.glove_mask[word_id, 0] = torch.tensor(1)
        #             if word in self.weight_set:
                        # self.glove_mask[word_id, 0] = torch.tensor(0)
        
            self.glove_mask = np.zeros((len(self.vocab)+4, 1))
            vocab_id_dict = self._index_token(self.vocab)
            for w in self.word_weight:
                if w in vocab_id_dict:                
                    word_id = vocab_id_dict[w]
                    self.glove_mask[word_id, 0] = torch.tensor(1)
                    
            self.vocab_id_dict = vocab_id_dict
        
            # convert id to vocab
            self.id_vocab_dict = {}
            for k, v in self.vocab_id_dict.items():
                self.id_vocab_dict[v] = k            
        
        else:
            
            self.glove_mask = np.ones((len(self.vocab)+4, 1))
            
    @staticmethod
    def _index_token(token_list):
        """
        # Generate index for the tokens used for the models.
        :return: dict with token as key and index as vakue
        """

        indexed_token_dict = dict()

        token_list_cp = list(token_list)  # un-change the original input
        
        token_list_cp.insert(0, 'UNK')    # for unknown tokens
        token_list_cp.insert(1, 'PAD')    # for unknown tokens
        
        for idx in range(len(token_list_cp)):
            indexed_token_dict[token_list_cp[idx]] = idx
        
        return indexed_token_dict
    
    def forward(self, input):
        return self.embedding(input)
    
    def _convert_token(self, token):
        if token not in self.vocab_id_dict:
            token = 'UNK'
        return self.vocab_id_dict[token]
    
    def word2index(self, words):
        """ convert words to id
        Args:
            words ([list]): tokenized word list
        """
        if isinstance(words, str):
            return self._convert_token(words)
        else:
            return [self._convert_token(w) for w in words]
    
    def _convert_id(self, id):
        if id not in self.id_vocab_dict:
            id = 0
        return self.id_vocab_dict[id]    
    
    def index2word(self, ids):
        """convert ids to words
        
        Args:
            ids (list): list of ids
        """
        if torch.is_tensor(ids):
            ids = ids.data.cpu().numpy().ravel().tolist()
        return [self._convert_id(id) for id in ids]
        
def topk_scatter(tensor, k, dim, device):
    """scatter top k values into one hot.
    Args:
        tensor ([type]): [description]
        k ([type]): [description]
        dim ([type]): [description]
    """
    tensor_device = tensor.device
    ans = torch.zeros(tensor.shape, device=tensor_device).requires_grad_(False)
    _, idx = torch.topk(tensor.float(), k, dim=dim)
    return ans.data.scatter_(dim, idx.type(torch.int64), 1.0).to(device)


def mojority_vote_predict(inputs, model, votes=10):
    """[summary]
    """
    
    batch_size = inputs[0].shape[0]
    preds_ = torch.zeros(batch_size,10, 2, device=inputs[0].device)
    for i in range(votes):
        pred_ = model(inputs)
        preds_[:, i, :] = pred_[:, -1, :]    
    return preds_.mean(1)

def loss_plot(folder, file, steps=0):
    
    fig = plt.figure(figsize=(12,6))

    address = os.path.join(folder, file, 'loss_log.json')
    batchs = []
    with open(address, 'r') as f:
        for line in f:
            b = json.loads(line)
            batchs.append(b)
    batchs_loss_pct = []

    for b in batchs:
        size = len(b['0'])
        if len(b['0']) != size:
            continue
        non_zero = np.array(b['0']) > 0.0
        loss_pct = []
        for step in range(steps):
            if step == 0:
                loss_pct.append(np.ones_like(np.array(b[str(step)])[non_zero]))
            else:
                loss_pct.append(np.array(b[str(step)])[non_zero]/np.array(b['0'])[non_zero])
        batchs_loss_pct.append(loss_pct)

    batch_mean_loss = np.zeros((len(batchs_loss_pct), steps))
    for i, b in enumerate(batchs_loss_pct):
        for step in range(steps):
            batch_mean_loss[i, step] = np.mean(b[step])
    plt.plot(batch_mean_loss.mean(0),)
        
    plt.savefig(os.path.join(folder, file, 'loss_plot.jpg'))
          
def custom_norm(tensor, item_num=None):
    """customized norm"""
    norm = torch.norm(tensor, p=1)
    if not item_num:
        shape = tensor.view(1,-1).shape[1]
        return norm/shape
    else:
        return norm/item_num

if __name__ == '__main__':
    from model.config_loader import config, path
    import spacy
    from model.data_loader import StockData
    from torch.utils.data import DataLoader
    from collections import defaultdict
    tag ={'<number>', 'UNK', 'PAD', 'pad', '<ticker>', '<date>', 'must'
          '.', ',', '?', '!', '>', '<', '+', '-', '=', 
          ';', '%', '-', '&', '_', ':', '*', '...', '✓', 'inc', 'via',
          'corp'}
    en = spacy.load('en_core_web_sm')
    stop_words = en.Defaults.stop_words
    avoid_set = tag.union(stop_words)
    # glove = GloveEmbedding(resource=path.glove, n_d=50, vocab_list=config['vocab'])
    # print(glove.index2word([0,1,2,3,4]))
    # print(glove.word2index(['<ticker>', '<number>']))
    # syner = Synonymer(glove, syn_num=15, syn_file=path.syn)
    # import torch, spacy
    
    # avoid_set = {'<number>', 'UNK', 'PAD', 'pad', '<ticker>', '<date>'}
    
    attack_layer = AttackMask(resource=path.glove, vocab_list=config['vocab'], avoid_set=avoid_set)
    # attack_layer.word2index(['<ticker>', '<number>'])
    # print(attack_layer(torch.tensor(attack_layer.word2index(['<ticker>', '<number>'])).long()))
    
    for corpus in ['all', 'fpb', 'adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
                 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance','science_fiction']:
        fin_words = WordWeight(resource=path.glove, vocab_list=config['vocab'], config={'attack_corpus': corpus}, stopwords=avoid_set)
    
    # tickers = []
    # for indu in config['stocks'].values():
    #     tickers += indu
    # # tickers = ['MSFT', 'JPM', 'DIS', 'GE']
    
    # dset = StockData(config['dates']['test'], tickers, 'cpu')
    # loader = DataLoader(dset, 1, shuffle=False)
    
    # vocab_test = defaultdict(int)
    
    # for i, inp in enumerate(loader, 0):
    #     # token_sum = torch.sum(inp[4][:, :, -1]).cpu().item()
    #     # fin_mask = fin_words(inp[0][:, :, -1].long())
    #     # fin_sum = torch.sum(fin_mask).cpu().item()
    #     # print('fin: {} | tot: {} | ratio: {}'.format(fin_sum, token_sum, fin_sum/token_sum) )
        
    #     tweet_amount = int(inp[1][0, 0, -1].cpu().item())
    #     for j in range(tweet_amount):
    #         tokens = fin_words.index2word(inp[0][0,0,-1,j,:])
    #         for t in tokens:
    #             vocab_test[t] += 1
        
    # with open('/home/yongxie2/AdvFinNLP/log/vocab_freq_test.json', 'w') as f:
    #     f.write(json.dumps(vocab_test))