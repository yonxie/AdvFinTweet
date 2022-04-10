import os
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

class BaseWrapper:
    def __init__(self, name, network, optimizer, votes, cfg):
        self.network = network
        self.optimizer = optimizer
        self.name = name
        self.votes = votes
        self.cfg = cfg
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

    def setup(self, is_train: bool):
        """set training or test mode"""
        if is_train:
            self.network.train()
        else:
            self.network.eval()

    def save_model(self, ckpt_path, epoch, save_optimizer=True):
        """save checkpoints to target directory"""

        if isinstance(self.network, DDP) or isinstance(self.network, DataParallel):
            module = self.network.module
        else:
            module = self.network
        state = {'epoch': epoch,
                 'state_dict': module.state_dict()}
        if save_optimizer:
            state['optim'] = self.optim.state_dict()
            state['optim_schedule'] = self.scheduler.state_dict()
        if not os.path.exists(os.path.join(ckpt_path, self.name)):
            os.mkdir(os.path.join(ckpt_path, self.name))
        if epoch != None:
            filename = os.path.join(ckpt_path, self.name, 'checkpoint_{}.pth.tar'.format(epoch))
            torch.save(state, filename)
        filename = os.path.join(ckpt_path, self.name, 'latest_checkpoint.pth.tar')
        torch.save(state, filename)
        
    def load_model(self, ckpt_path, epoch=None, load_optimizer=True):
        """load model parameters
        
        Args:
            ckpt_path: directory
            epoch: the epoch the load. Load most recent parameter if None
            load_optimizer: if load optimizer
        return: epoch of checkpoint
        """
        if epoch != None:
            filename = os.path.join(ckpt_path, self.name, 'checkpoint_{}.pth.tar'.format(epoch))
        else:
            filename = os.path.join(ckpt_path, self.name, 'latest_checkpoint.pth.tar')
            
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            # checkpoint['state_dict'].pop('embedding.embedding.weight')
            if isinstance(self.network, DDP) or isinstance(self.network, DataParallel):
                missing, _ = self.network.module.load_state_dict(checkpoint['state_dict'])
            else:
                missing, _= self.network.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded network checkpoint '{}'".format(filename))
            print('=> missing keys: {}', missing)
            if load_optimizer:
                self.optim.load_state_dict(checkpoint['optim'])
                self.scheduler.load_state_dict(checkpoint['optim_schedule'])
                print("=> loaded optimizer and scheduler checkpoints '{}'".format(filename))
            return checkpoint['epoch'] + 1
        else:
            raise Exception("=> no checkpoint found at '{}'".format(filename))

    def zero_grad(self):
        self.network.zero_grad()
        
    def __call__(self, inputs, attack=None, mode='normal_forward', votes=None, **kwargs):
        # self.setup(is_train=0)
        if not votes:
            votes = self.votes
        return self.network(inputs, attack, mode, votes=votes, **kwargs)
        
    def eval(self):
        self.setup(is_train=False)
    
    def train(self):
        self.setup(is_train=True)
        
    def run(self, loader, train, interval=20):
        """Train model on given data loader.
        
        Args:
            loader: data loader
            train: bool
            interval: niter to report statistics
        Returns:
        """
        
        stage = 'train' if train else 'test'

        self.setup(is_train=train)
        criterion = nn.CrossEntropyLoss()
        
        tot_count, tot_loss, pos_count = 0, [], 0
        
        if stage == 'train':
            self.scheduler.step()
        
        pred_labels, true_labels = torch.zeros([1]), torch.zeros([1])
        
        for i, inputs in enumerate(loader, 0):
            
            if inputs[0].shape[0] == 1: continue         
            
            y_p = self.network(inputs, votes=self.votes)[0]
            loss = criterion(y_p[:,-1,:], inputs[-1][:,0,-1].long())
            
            # backward loss
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # ctralculate evaluation metric
            pred_label = torch.argmax(y_p[:,-1], dim=1).cpu()
            true_label = inputs[-1][:,-1, -1].cpu()
  
            pred_labels = torch.cat([pred_labels, pred_label], dim=0)
            true_labels = torch.cat([true_labels, true_label], dim=0)
            tot_f1 = f1_score(true_labels[1:], pred_labels[1:], pos_label=1)
            tot_acc = accuracy_score(true_labels[1:], pred_labels[1:], normalize=True)
            
            tot_loss.append(loss.cpu().item())
            pos_count = torch.sum(pred_labels[1:]).cpu().item()
            tot_count = pred_labels.shape[0] - 1
            
            if i % interval in [0] and i > 1:                                
                print('=> stage: {} | niter: {} | Loss: {:.4f} | Acc: {:.4f} | F1: {:.4f} | Pos: {:.4f}'.format(stage, i, np.nanmean(tot_loss), tot_acc, tot_f1, pos_count/tot_count))
                 
        return tot_acc, tot_f1, pos_count/tot_count, np.nanmean(tot_loss)