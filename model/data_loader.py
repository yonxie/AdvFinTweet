# -*- coding: utf-8 -*-
"""
@file: data_loader.py
@author: Yong Xie
@time: 5/12/2021 11:17 AM
@Description: 

# This script process textual data into embeddings.

"""

import os
import json
import numpy as np
import torch
import sys
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

sys.path.append('..')
from model.config_loader import path, config, logger


class StockData(Dataset):
    """
    # A textual-numerical Dataset inherited from Pytorch Dataset.
    It generate the feature and label of multi-stock for consecutive dates.

    # It doesn't embed words, but only ids for corresponding embedding method.
    So word/sentence embedding should be used in the model.

    # Output:
        text: token ids for s specific embedding, in the shape of (stock, day, message, word)
        message_count: number of message of each stock for each day, shape (stock, day)
        word_count: number of words of each message of each stock for each day, shape (stock, day, message)
        technical: technical features, in the shape of (stock, day, feature)
        label: label, in the shape of (stock, day)

    """

    def __init__(self, dates, tickers, device):

        # load path
        self.technical_path = path.technical
        self.text_path = path.text
        self.label_path = path.label
        self.vocab_path = path.vocab
        self.glove_path = path.glove

        # load dates
        self.data_dates = dates 
        self.global_start_date = '2014-01-01'
        self.global_end_date = '2016-01-01'

        self.day_step = config['model']['day_step']
        self.day_shuffle = config['model']['day_shuffle']
        self.ticker_combine = config['model']['ticker_combine']  # if combine all the tickers for each day
        self.ticker_shuffle = config['model']['ticker_shuffle']  # if shuffle tickers if not combined
        self.text_combine = config['model']['text_combine']      # if combine text for each day

        self.max_n_words = config['model']['max_n_words']
        self.max_n_msgs = config['model']['max_n_msgs']
        self.tech_feature_size = config['model']['tech_feature_size']
        self.tech_feature_size = 3

        self.word_embed_type = config['model']['word_embed_type']
        self.word_embed_size = config['model']['word_embed_size']
        self.stock_embed_size = config['model']['stock_embed_size']
        self.device = device

        self.tickers = tickers
        self._gen_dates()

        self.label = self._read_return()
        self.text, self.words_count, self.msgs_count, self.msgs = self._read_text()
        self.technical = self._read_technical()

        self.index = self._generate_index()

    def _gen_dates(self):
        """
        # Generate the list of training tickers and dates
        """

        self.dates = defaultdict(list)
        missing = []
        for ticker in self.tickers:
            file = pd.read_csv(os.path.join(self.technical_path, ticker+'.csv'), parse_dates=[0])
            dat = pd.date_range(self.global_start_date, self.global_end_date, freq='D')
            dates = file['date'][(file.date >= dat[0]) & (file.date <= dat[-1])]
            if len(dates) > len(dat) * config['train']['min_date_ratio']:
                self.dates[ticker] = [date.strftime('%Y-%m-%d') for date in dates]
            else:
                missing.append(ticker)
        for miss in missing:
            self.tickers.remove(miss)
            
    def _read_text(self):
        """
        # Read text data in json format, and then return list of strings.
        :return: dict of text, each for a stock on a day

        """
        if self.word_embed_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if self.word_embed_type == 'glove':
            token_id_dict = self._index_token(config['vocab'])

        texts, words_count, msgs_count, msgs = {}, {}, {}, {}
        for tic in self.tickers:
            missing = []
            texts[tic], words_count[tic], msgs_count[tic], msgs[tic] = {}, {}, {}, defaultdict(list)
            for date in self.dates[tic]:
                tic_date_text = np.zeros([self.max_n_msgs, self.max_n_words])
                num_words = np.ones([self.max_n_msgs, ], dtype=int)
                file = os.path.join(self.text_path, tic, date)
                msg_count = 0
                try:
                    with open(file, 'r') as f:
                        for line in f:
                            msg_dict = json.loads(line)
                            text = msg_dict['text'][:config['model']['max_n_words']]                # the text is already tokenized
                            if len(text) == 0:
                                print('no text:', text)
                                continue
                            msgs[tic][date].append(text)
                            if self.word_embed_type == 'bert':
                                indexed_tokens = tokenizer.convert_tokens_to_ids(text)
                                num_words[msg_count] = int(min(len(indexed_tokens), self.max_n_words))
                                tic_date_text[msg_count, :num_words[msg_count]] = indexed_tokens[:num_words[msg_count]]
                            elif self.word_embed_type == 'glove':
                                indexed_tokens = self._token2index(text, token_id_dict)
                                num_words[msg_count] = len(indexed_tokens)
                                if len(indexed_tokens) == 0:
                                    print('zero length:', indexed_tokens)
                                tic_date_text[msg_count, :len(indexed_tokens)] = indexed_tokens
                            else:
                                raise Exception('unsupported embedding type!')
                            msg_count += 1
                            if msg_count >= self.max_n_msgs:
                                break
                    if msg_count == 0:
                        missing.append(date)
                        continue
                    texts[tic][date] = tic_date_text
                    words_count[tic][date] = num_words
                    msgs_count[tic][date] = msg_count
                except Exception as e:
                    missing.append(date)
            for miss in missing:
                self.dates[tic].remove(miss)
        return texts, words_count, msgs_count, msgs

    def _read_technical(self):
        """Read technical features."""
        # collect data
        technical = {}
        for tic in self.tickers:
            file = pd.read_csv(os.path.join(self.technical_path, '{}.csv'.format(tic.upper())), parse_dates=[0])
            file['date'] = file['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            file = file[['date', 'high', 'low', 'close']]
            technical[tic] = file.set_index('date').T.to_dict('list')
        return technical

    def _read_return(self):
        """Read label, here returns for each tickers."""
        ret = {}
        for tic in self.tickers:
            file = pd.read_csv(os.path.join(self.label_path, '{}.csv'.format(tic.upper())), parse_dates=[0])
            file['date'] = file['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            ret[tic] = file.set_index('date').T.to_dict('list')
        return ret

    def _generate_index(self):
        """
        # Generate sample index for dates and tickers, by taking steps into account.
        # If stock are combined, then only index dates; otherwise, index both dates
        # and tickers
        
        """

        dataset_dates = []
        for start, end in self.data_dates:
            dataset_dates += [date.strftime('%Y-%m-%d') for date in pd.date_range(start, end, freq='D')]
            
        index = {}
        if self.ticker_combine:
            dates = [i for i in range(self.day_step, len(self.dates))]     # TODO: add ticker list
            if self.day_shuffle:
                np.random.shuffle(dates)
            idx = 0
            for d in dates:
                index[idx] = d
                idx += 1
        else:
            tickers = [i for i in range(len(self.tickers))]
            if self.ticker_shuffle:
                np.random.shuffle(tickers)
            idx = 0
            for t in tickers:
                ticker = self.tickers[t]
                dates = [i for i in range(self.day_step, len(self.dates[ticker]))]
                if self.day_shuffle:
                    np.random.shuffle(dates)
                for d in dates:
                    if self.dates[ticker][d] not in dataset_dates:
                        continue
                    if config['model']['threshold']:
                        if 0.0055 > self.label[ticker][self.dates[ticker][d]][0] >= -0.005:
                            continue
                    index[idx] = [[t], d]
                    idx += 1

            ticker_info = {}
            for t in tickers:
                ticker_info[t] = self.tickers[t]
            
            stock_info = {'dates': self.dates, 'tickers':ticker_info}
            
            with open('ticker_date_index_dict.json', 'w') as f:
                f.write(json.dumps(stock_info))
                        
        return index

    @staticmethod
    def _index_token(token_list):
        """
        # Generate index for the tokens used for the models.
        :return: dict with token as key and index as vakue
        """

        indexed_token_dict = dict()

        token_list_cp = list(token_list)  # un-change the original input
        token_list_cp.insert(0, 'UNK')    # for unknown tokens
        token_list_cp.insert(1, 'PAD')    # for Pad tokens
        
        for idx in range(len(token_list_cp)):
            indexed_token_dict[token_list_cp[idx]] = idx

        return indexed_token_dict

    @staticmethod
    def _token2index(tokens, vocab_index_dict):
        """
        # Convert tokens to index in the vocabulary dict
        :param tokens: list of tokens
        :param vocab_index_dict: token:index pair
        :return: index list
        """
        def _convert_token(token):
            if token not in vocab_index_dict:
                token = 'UNK'
            return vocab_index_dict[token]

        return [_convert_token(w) for w in tokens]
        
    def __getitem__(self, idx):
        """
        :param idx:
        :return: data for that index, in the shape of (stock, date, data)

        """
        tickers, end_date = self.index[idx]
        text = torch.ones([len(tickers), self.day_step, self.max_n_msgs, self.max_n_words], device=self.device)
        words_count = torch.ones([len(tickers), self.day_step, self.max_n_msgs], dtype=torch.int, device=self.device)
        msgs_count = torch.zeros([len(tickers), self.day_step], dtype=torch.int, device=self.device)
        msgs_mask = torch.zeros([len(tickers), self.day_step, self.max_n_msgs], device=self.device)
        words_mask = torch.zeros([len(tickers), self.day_step, self.max_n_msgs, self.max_n_words], device=self.device)
        technical = torch.zeros([len(tickers), self.day_step, self.tech_feature_size], device=self.device)
        label = torch.zeros([len(tickers), self.day_step], device=self.device)
        return_label = torch.zeros([len(tickers), self.day_step], device=self.device)
        for t, tic_idx in enumerate(tickers):
            tic = self.tickers[tic_idx]
            offset = 0
            for d in range(self.day_step):
                text[t, d, :, :] = torch.tensor(self.text[tic][self.dates[tic][end_date-self.day_step+d+offset]], device=self.device)
                words_count[t, d, :] = torch.tensor(self.words_count[tic][self.dates[tic][end_date-self.day_step+d+offset]], device=self.device).int()
                msgs_count[t, d] = torch.tensor(self.msgs_count[tic][self.dates[tic][end_date-self.day_step+d+offset]], device=self.device).int()
                technical[t, d, :] = torch.tensor(self.technical[tic][self.dates[tic][end_date-self.day_step+d+offset]], device=self.device)
                label[t, d] = torch.tensor(self.label[tic][self.dates[tic][end_date-self.day_step+d+1]][0] > 0, device=self.device)
                return_label[t, d] = torch.tensor(self.label[tic][self.dates[tic][end_date-self.day_step+d+1]][0], device=self.device)
                msgs_mask[t, d, :msgs_count[t, d]] = 1.0
                for m in range(self.max_n_msgs):
                    words_mask[t, d, m, :words_count[t, d, m]] =  1.0   
        msgs_count = torch.sum(msgs_mask, dim=2)
        
        return [text, msgs_count, msgs_mask, words_count, words_mask, technical, label.int()]

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    
    from model.config_loader import config, path
    tickers = []
    for indu in config['stocks'].values():
        tickers += indu
        
    test_dset = StockData(config['dates']['train'], tickers, 'cpu')
    print('test:', test_dset.__len__())
    
    tot_msg_count = 0
    for tic in test_dset.msgs_count:
        for date in test_dset.msgs_count[tic]:
            tot_msg_count += test_dset.msgs_count[tic][date]

    print('total tweets:', tot_msg_count)
    