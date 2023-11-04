from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from dataloaders.dataloader import Custom_dataset
import torchvision.datasets as datasets

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)
        self.mem_ova = {}
        self.mem_raw = {}
        self.mem_q = {}
        self.mem_count = 0
 
    def update_model(self, inputs, targets, task=None, ortho=False, maml=False):

        if len(self.mem_q) == 0:
            memQ = None
        else:
            memQ = self.mem_q
            
        # logits
        if ortho:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True, ortho_n_fix=True, memQ=memQ)
        elif maml:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True, maml=True, memQ=memQ)
        else:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.clone().long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        ova_loss = 0
        if task != None:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True)
            self.update_mem_ova(latent, targets, task)
            self.update_mem_raw(inputs, targets, task)
            self.update_mem_q(q, targets, task)
            ova_loss = self.compute_ovaLoss0()
            self.optimizer.zero_grad()
            ova_loss.backward()
            self.optimizer.step()
        
        total_loss += ova_loss
        
        return total_loss.detach(), logits
    
    def update_model0(self, inputs, targets, task=None, ortho=False):

        if len(self.mem_q) == 0:
            memQ = None
        else:
            memQ = self.mem_q
            
        # logits
        if ortho:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True, ortho_n_fix=True, memQ=memQ)
        else:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.clone().long(), dw_cls)
        ova_loss = self.compute_ovaLoss(latent, targets, task[0])

        # ce loss
        total_loss = total_loss + prompt_loss.sum() + ova_loss

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        ova_loss = 0
        if task != None:
            logits, prompt_loss, latent, q = self.model(inputs, train=True, get_latent=True)
            self.update_mem_ova(latent, targets, task)
            self.update_mem_raw(inputs, targets, task)
            self.update_mem_q(q, targets, task)
        
        total_loss += ova_loss
        
        return total_loss.detach(), logits
    

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters()) + list(self.OOD_heads.module.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters()) + list(self.OOD_heads.parameters())
        # print('params to opt', params_to_opt)
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.OOD_heads = self.OOD_heads.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
            self.OOD_heads = torch.nn.DataParallel(self.OOD_heads, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self
    
# Our method <3
class OVAPrompt(Prompt):

    def __init__(self, learner_config):
        super(OVAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'ovap',prompt_param=self.prompt_param)
        return model

# @InProceedings{Smith_2023_CVPR,
#     author    = {Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#     title     = {CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free Continual Learning},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2023},
#     pages     = {11909-11919}
# }
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model
    
class NoP(NormalNN):

    def __init__(self, learner_config):
        super(NoP, self).__init__(learner_config)

    # def create_model(self):
    #     cfg = self.config
    #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'None',prompt_param=self.prompt_param)
    #     return model