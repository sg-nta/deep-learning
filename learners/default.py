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
import copy
import torchvision
from utils.schedulers import CosineSchedule
from utils.ova import ova_loss
from utils.mixup import mixup_data, mixup_criterion
from dataloaders.dataloader import Custom_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import PIL.Image
from torchvision.transforms import ToTensor
import pandas as pd
from scipy.stats import wasserstein_distance
#import ot

import sklearn
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sb
from glob import glob
import plotly.express as px

from utils.ova import feat_bootleneck_rf as feat_bootleneck_rf

class OOD_heads(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(OOD_heads, self).__init__()
        
        self.fc_ood=nn.Linear(input_size, num_classes*2, bias=False)
        self.norm = norm
        self.tmp = temp
            
    def forward(self, z):
        x = z
        if self.norm:
            x = F.normalize(x)
            x = self.fc_ood(x)/self.tmp
        else:
            x = self.fc_ood(x)
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)
        
class OOD_heads2(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(OOD_heads, self).__init__()
        self.fc_pre = nn.Linear(input_size, 2048)
        # self.fc_pre = feat_bootleneck_rf(bottleneck_dim=input_size, nrf=2048)
        self.fc_ood=nn.Linear(2048, num_classes*2, bias=False)
        self.norm = norm
        self.tmp = temp
            
    def forward(self, z):
        x = F.relu(self.fc_pre(z))
        if self.norm:
            x = F.normalize(x)
            x = self.fc_ood(x)/self.tmp
        else:
            x = self.fc_ood(x)
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)
        
class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        print(self.config)
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.OOD_heads = OOD_heads(self.out_dim, 768) #
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']
        self.num_cls_per_task = int(self.config['num_classes']/ self.config['num_tasks'])

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0
        # self.mem_ova = {}

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            
            # data weighting
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    if self.config['learner_name'] == 'OVAPrompt' and task[0] > 0:
                        if epoch < self.config['maml_e']:
                            # learn k without any constraints (task0) or maml opt
                            xxx = self.update_model(x, y, task=task, maml=True)
                        else:
                            # ortho project + fix k, just learn p  
                            xxx = self.update_model(x, y, task=task, ortho=True)
                    else:
                        xxx = self.update_model(x, y, task=task)

                    if len(xxx) == 3:
                        loss, output, latent = xxx
                        
                    else:
                        loss, output = xxx

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        # self.viz_latent_space_final()
        # self.viz_latent_shift()

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None
        
        
    def finetune_OVA(self, model_save_dir):
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass
            
        print('>>>>> Fine tune OVA head at final....')
        for iter in range(1001):
            ova_loss = self.compute_ovaLoss0()
            
            self.optimizer.zero_grad()
            ova_loss.backward()
            self.optimizer.step()
            
            if iter % 100 == 0:
                print('OVA loss: ', ova_loss.item())
            
    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 
    
    def compute_ovaLoss0(self):
        # if mem
        loss_open_a = 0
        loss_open_b = 0
        lambQ = 0.6 #put on pos constraints 
        for i in self.mem_ova.keys():
            z, y = self.mem_ova[i]
            # z, ya, yb, lamb = mixup_data(z, y, 1.0, use_cuda=True)
            ya = y
            out_ood = self.OOD_heads(z)
            out_ood = out_ood.view(z.size(0), 2, -1)
            open_loss_pos_a, open_loss_neg_a = ova_loss(out_ood, ya)
            # loss_open_a += 0.5 * (open_loss_pos_a + open_loss_neg_a)
            loss_open_a += lambQ * open_loss_pos_a + (1-lambQ) * open_loss_neg_a
            
            # open_loss_pos_b, open_loss_neg_b = ova_loss(out_ood, yb)
            # loss_open_b += 0.5 * (open_loss_pos_b + open_loss_neg_b)
            # loss_open_b += lambQ * open_loss_pos_b + (1-lambQ) * open_loss_neg_b
            
        loss_open = loss_open_a #* lamb + loss_open_b * (1 - lamb)
        return loss_open
    
    def compute_ovaLoss(self, z_t, y_t, t):
        # if mem
        loss_open_a = 0
        loss_open_b = 0
        lambQ = 0.6 #put on pos constraints 
        
        for i in self.mem_ova.keys():
            if i >= t:
                z, y = z_t, y_t 
            else:
                z, y = self.mem_ova[i]
            # z, ya, yb, lamb = mixup_data(z, y, 1.0, use_cuda=True)
            ya = y
            out_ood = self.OOD_heads(z)
            out_ood = out_ood.view(z.size(0), 2, -1)
            open_loss_pos_a, open_loss_neg_a = ova_loss(out_ood, ya)
            # loss_open_a += 0.5 * (open_loss_pos_a + open_loss_neg_a)
            loss_open_a += lambQ * open_loss_pos_a + (1-lambQ) * open_loss_neg_a
            
            # open_loss_pos_b, open_loss_neg_b = ova_loss(out_ood, yb)
            # loss_open_b += 0.5 * (open_loss_pos_b + open_loss_neg_b)
            # loss_open_b += lambQ * open_loss_pos_b + (1-lambQ) * open_loss_neg_b
            
        loss_open = loss_open_a #* lamb + loss_open_b * (1 - lamb)
        return loss_open
    
    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None, task=None):
        
        #update model with CE loss 
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits, latent = self.forward(inputs, get_latent=True)
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        
        ova_loss = 0
        #update features for ova loss after update
        if task != None:
            self.update_mem_raw(inputs, targets, task)
            ova_loss = self.compute_ovaLoss()
            
        total_loss += ova_loss   
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits, latent
    
    def update_mem_ova(self, latent, targets, task, num_samples=100): #numsample per class
        self.mem_ova[int(task[0])] = (latent[:num_samples].detach(), targets[:num_samples].detach())

    def update_mem_raw(self, sample, targets, task, num_samples=50):
        self.mem_raw[int(task[0])] = (sample[:num_samples].cpu(), targets[:num_samples].cpu())
        
    def update_mem_q(self, q, targets, task, num_samples=1600):
        
        if len(self.mem_q) <= task[0]:
            self.mem_q[int(task[0])] = (q[:num_samples].detach(), targets[:num_samples])
        else: 
            remain = num_samples - len(self.mem_q[int(task[0])][0])
            if remain > 0:
                temp_q, temp_y = self.mem_q[int(task[0])]
                temp_q = torch.cat((temp_q, q[:remain]), dim=0)
                temp_y = torch.cat((temp_y, targets[:remain]), dim=0)
                self.mem_q[int(task[0])] = (temp_q.detach(), temp_y.detach())
            
    
    
    def viz_latent_space_final(self):
        ### collect feature vectors of classes
        features = torch.empty(0).cuda()
        labels = torch.empty(0).cuda()
        for z_t in self.mem_ova.values():
            fea, lab = z_t
            features = torch.cat((features, fea), dim=0)
            labels = torch.cat((labels, lab), dim=0)
        
        try:  
            self.main_centroids = self.model.module.last.weight.data
        except:
            self.main_centroids = self.model.last.weight.data
        
        # print(features.shape, labels.shape)
        self.viz_cluster(features, labels)
        
        
    def viz_latent_shift(self):
        
        old_features = torch.empty(0).cuda()
        for z_t in self.mem_ova.values():
            fea, lab = z_t
            old_features = torch.cat((old_features, fea), dim=0)
        old_labels = [0]*len(old_features)
        old_labels = torch.LongTensor(old_labels)
        
        curr_features = torch.empty(0).cuda()
        for i, sub in enumerate(self.mem_raw.values()):
            x, y = sub
            if i == 0:
                temp_set = Custom_dataset(x[:100], y[:100])
                
            else:
                temp_set.add_new(x, y)
        
        self.model.eval()
        raw_loader = DataLoader(temp_set, batch_size=64, num_workers=self.config['n_workers'])
        for inp, lab in raw_loader:
            if self.gpu:
                inp = inp.cuda()
            _, latent = self.model.forward(inp, get_latent=True)
            
            curr_features = torch.cat((curr_features, latent.detach()), dim=0) 
        curr_labels = [100]*len(curr_features)
        curr_labels = torch.LongTensor(curr_labels)
        
        # print("Ws distance... between old and new features:")
        # # print(wasserstein_distance(old_features.cpu(), curr_features.cpu()))
        # print(ot.emd(old_features, curr_features, torch.ones(old_features.shape[0], curr_features.shape[0])))
        
        features = torch.cat((old_features, curr_features), dim=0)
        labels = torch.cat((old_labels, curr_labels), dim=0)
        
        self.viz_cluster(features, labels, mode2=True)

        
    def plot(self, x, colors, data_len):
  
        palette = np.array(sb.color_palette("hls", 201))  #Choosing color palette 

        # Create a scatter plot.
        f = plt.figure()
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=100, c=palette[colors.astype(np.int32)])
        # ax2 = plt.axes()
        
        # f = px.scatter_3d(pd.DataFrame(x.reshape(3, -1)), x=0, y=1, z=2,) 
        
        # Add the labels for each digit.
        txts = []
        # for i in colors:
        #     # Position of each label.
        #     xtext, ytext = np.mean(x[colors == i, :], axis=0)
        #     txt = ax.text(xtext, ytext, str(int(i)), fontsize=15)
        #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        #     txts.append(txt)
        
        # for i in colors:
        #     # Position of each label.
        #     xtext, ytext = np.mean(x[colors == i, :], axis=0)
        #     txt = ax.text(xtext, ytext, str(int(i)), fontsize=15)
        #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        #     txts.append(txt)
            
        # sc = ax.scatter(x[data_len:,0], x[data_len:,1], s=80, c='r', marker='x')
        return f, ax, txts
    
    def viz_cluster(self, features, labels, mode2=False): 
               
        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        centroids_ = self.main_centroids.cpu().detach().numpy()
        labels_ = torch.Tensor(range(len(self.main_centroids))).cpu().detach().numpy()
                
        labels_ = np.array(labels_)
        len_data = features.shape[0]
        data_all = features[labels%10==0]# np.vstack([features, centroids_])
        label_all = labels[labels%10==0]# np.hstack([labels, labels_])
        tsneX = TSNE(n_components=2,perplexity=5, init='pca').fit_transform(data_all)
        f, ax, txt =  self.plot(tsneX, label_all, len_data)
        
        # exp.log_figure(f'InitStep_{args.batch_id}', f)
        save_dir = self.config['log_dir']
        if mode2:
            # f.savefig(f'{save_dir}_Slatent_{self.task_count}.jpg')
            # f = f.ToTensor()
            # image = PIL.Image.open(f)
            # image = ToTensor()(image).unsqueeze(0)
            # f_grid = torchvision.utils.make_grid(f) ax2
            writer.add_figure(f'{save_dir}_Slatent_{self.task_count}.jpg', f)
        else:
            # f.savefig(f'{save_dir}_latent_{self.task_count}.jpg')
            # canvas = f.canvas
            # ax = f.gca()
            # canvas.draw() 
            # image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8') 
            # image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
            writer.add_figure(f'{save_dir}_latent_{self.task_count}.jpg', f)
            
        
        
    def test_ood(self, data, target, curr_task_id, real_task_id):
        total_loss = 0
        total_num_pos = 0
        correct_pos = 0
        
        pos_score = torch.zeros((curr_task_id + 1)).cuda()
        pos_total_score = torch.zeros((curr_task_id + 1)).cuda()
        pos_num_score = torch.zeros((curr_task_id + 1)).cuda()
        
        pos_hit_score = torch.zeros((curr_task_id + 1)).cuda()
        pos_hit_total_score = torch.zeros((curr_task_id + 1)).cuda()
        
        self.model.eval()
        self.OOD_heads.eval()
        
        with torch.no_grad():
            # Loop batches
            
                data, target = data.cuda(), target.cuda()
                output, features = self.forward(data, get_latent=True)
                
                out_ood = F.softmax(self.OOD_heads(features).view(features.size(0), 2, -1), 1)
                max_scores = out_ood[:, 1, :]
                for t in range(curr_task_id):
                    pos_total_score[t] += max_scores[:, t*self.num_cls_per_task:t*self.num_cls_per_task+self.num_cls_per_task].max(dim=1)[0].sum()
                    pos_num_score[t] += out_ood.shape[0]
                    pos_score[t] = pos_total_score[t]/pos_num_score[t]
                        
                print('pos score - ', pos_score[:curr_task_id])
        
        return pos_score
            

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False):

        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):
            
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
                    
            # rough --> to viz
            # logits, prompt_loss, latent, q = self.model(input, train=True, get_latent=True)
            # self.update_mem_ova(latent, target, task)
            # self.update_mem_raw(input, target, task)
            # self.viz_latent_space_final()
            # self.viz_latent_shift()
            
            mulOOD = None
            if len(input) > 1:
                ood_score = self.test_ood(input, target, self.task_count, task[0])
                mulOOD = ood_score.view(-1, 1).repeat(1, self.num_cls_per_task).view(-1)  

            else:
                print('Len input', len(input))
            
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                if mulOOD != None:
                    mulOOD = mulOOD
                    output = output * mulOOD[:self.valid_out_dim]
                    
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if task_global:
                        output = model.forward(input)[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        output = model.forward(input)[:, task_in]
                        acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        
        ova_state = self.OOD_heads.state_dict()
        for key in ova_state.keys():  # Always save it to cpu
            ova_state[key] = ova_state[key].cpu()
        self.log('=> Saving class ova to:', filename)
        torch.save(ova_state, filename + 'ova.pth')
        
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        # self.log('=> Load Done')
        
        self.OOD_heads.load_state_dict(torch.load(filename + 'ova.pth'))
        self.log('=> Load Done')
        
        if self.gpu:
            self.model = self.model.cuda()
            self.OOD_heads = self.OOD_heads.cuda()
        self.model.eval()
        self.OOD_heads.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':[self.model.parameters(), self.OOD_heads.parameters()],
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
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x, get_latent=False):         
        out, latent = self.model.forward(x, get_latent=get_latent)
        out = out[:, :self.valid_out_dim]
        return out, latent

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

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

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.manifold import TSNE
import time
def plot(x_tsne, d_y,):
    
        palette_test = np.array(sb.color_palette("tab10", 10))  #Choosing color palette 

        # Create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        x_test = x_tsne        
        y_test = d_y.cpu().numpy()
        
        try: 
            sc = ax.scatter(x_test[:,0], x_test[:,1], lw=0, s=40, c=palette_test[y_test.astype(np.int64)], label='t'+str(y_test.astype(np.int64)), marker='^')

        except:
            palette_batch = np.array(sb.color_palette("tab10", 100))  #Choosing color palette 
            palette_test = np.array(sb.color_palette("tab10", 100))  #Choosing color palette 
            sc = ax.scatter(x_test[:,0], x_test[:,1], lw=0, s=40, c=palette_test[y_test.astype(np.int)], label='t'+str(y_test.astype(np.int64)), marker='^')
            
        txts = None

        return f, ax, txts
        
                
def viz_cluster(features, targets):
        
            tsne_in = features.cpu()
            len_test = len(tsne_in)
        
            d_y = targets
            d_yunique = torch.unique(d_y).squeeze()

            if True:
                tsne = TSNE()
                x_tsne = tsne.fit_transform(tsne_in.detach().numpy())
            # except:
            #     print('test', f_x.isnan().any())
            #     print('train', f_ext.isnan().any())
            #     print('centroids', centroids_.isnan().any())
            
            file_name = f'Time_{time.time()}'
            f, ax, txts = plot(x_tsne, d_y)
            
            f.savefig(file_name)
            
            return f, ax, txts, file_name