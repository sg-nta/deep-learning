import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy

def att_score(m_input, k_per_task):
    """
    m_inputs: b * k
    """
    # print(m_input)
    
    shape = m_input.shape
    temp = m_input.view(shape[0], -1, k_per_task)
    temp = torch.abs(temp)
    # print(temp)
    sum_temp = temp.sum(dim=2)
    
    # print(sum_temp)
    
    m = nn.Softmax(dim=1)
    sum_temp = m(sum_temp)
    
    mean_temp = sum_temp.mean(dim=0)
    
    
    # print(temp.shape, sum_temp.shape, mean_temp.shape)
    print(mean_temp)
    # exit()

class OVAPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.ortho_prj_once = False
        self.ortho_prj_twice = False
        self.threshold = prompt_param[-1]

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence

            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)
            
        self.ortho_prj_once = False
        # self.constraints_k()

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 
    
    # code for this function if modified from:
    # https://github.com/sahagobinda/GPM/blob/main/main_pmnist.py
    def ortho_projection(self, keys, subspace):
        subspace = torch.Tensor(subspace)
        subspace = subspace.cuda()
        keys_new = keys - torch.mm(keys, torch.mm(subspace, subspace.t()).detach())
        return keys_new
    
    def compute_subspace(self, memQ, threshold=0.975):
        rep_matrix = torch.empty(0)
        if True:
            for i in memQ.keys():
                q, y = memQ[i]
                rep_matrix = torch.cat((rep_matrix, q.cpu()), dim=0)
            
        rep_matrix = rep_matrix.t().cpu().clone()
            
        #compute subspace:
        U,S,Vh = torch.linalg.svd(rep_matrix, full_matrices=False)
        sval_total = (S**2).sum()
        sval_ratio = (S**2)/sval_total
        r = torch.sum(np.cumsum(sval_ratio)<threshold) #+1  
        subspace = U[:,0:r].detach()
        return subspace
    
    def constraints_k(self, memQ, threshold=0.975):
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        
        subspace_byQ = self.compute_subspace(memQ, threshold=threshold)
        
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            k = K.clone().detach()
            k_curr = k[s:f]
            k_curr = self.ortho_projection(k_curr, subspace_byQ).detach()
            k[s:f] = k_curr
            setattr(self, f'e_k_{e}',k)
    
    def forward(self, x_querry, l, x_block, train=False, task_id=None, ortho_n_fix=False, maml=False, memQ=None): #use weight and ortho prj
        
        if ortho_n_fix and self.ortho_prj_once == False and memQ != None: # have not constraint K
            #option 1: ortho prj just once and then fix keys
            self.constraints_k(memQ)
            self.ortho_prj_once = True
        elif maml and self.ortho_prj_twice == False and memQ != None:    
            #option 2: compute subspace once at starting point of each task ---> use to optimize k
            self.subspace_byQ = self.compute_subspace(memQ, threshold=self.threshold)
            self.ortho_prj_twice = True

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            
            #option 1: 
            if ortho_n_fix:
                K = K.detach()
            
            elif maml:
                #option 2: 
                K = self.ortho_projection(K, self.subspace_byQ.detach())
    
            n_K = nn.functional.normalize(K, dim=1)
            n_q = nn.functional.normalize(x_querry, dim=1)
            q_k = torch.einsum('bd,kd->bk', n_q, n_K)
            
            # att_score(q_k, pt)
            # # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', q_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes, bias=False)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'ovap':
            self.prompt = OVAPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False, get_latent=False, ortho_n_fix=False, maml=False, memQ=None):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, ortho_n_fix=ortho_n_fix, maml=maml, memQ=memQ)
            out = out[:,0,:]

        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
            
        latent_vectors = out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            if get_latent:
                return out, prompt_loss, latent_vectors, q
            else:
                return out, prompt_loss
        elif get_latent:
            return out, latent_vectors
        else:
            return out
            
def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)



