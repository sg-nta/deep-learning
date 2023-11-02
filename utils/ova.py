import torch
import torch.nn.functional as F
import torch.nn as nn

class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner

class RandomFourierFeatures(FeatureMap):
    def __init__(self, query_dimensions, n_dims=None, gamma=0.5):
        super(RandomFourierFeatures, self).__init__(query_dimensions)

        self.n_dims = n_dims
        self.gamma = gamma

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            "omega",
            torch.zeros(query_dimensions, self.n_dims//2)
        )

    def new_feature_map(self):
        self.omega.normal_()

    def forward(self, x):  # x ~ 10^-2
        # print(x.shape)
        # x = x * math.sqrt(self.softmax_temp)
        x = x * self.gamma  # gamma = 0.1 => x ~ 10^-3
        u = x.matmul(self.omega)  # omega ~ 10^-3 * x (~10^-3), u (bs, nrf//2) ~ 10^-1
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)  # phi (bs, nrf) ~ 10^-1
        # return phi * math.sqrt(2/self.n_dims)
        return phi
    
class feat_bootleneck_rf(nn.Module):
    def __init__(self, gamma = 0.5, bottleneck_dim=256, type="ori", nrf = 512):
        super(feat_bootleneck_rf, self).__init__()
        self.bn2 = nn.BatchNorm1d(nrf, affine=True)
        self.type = type

        self.feature_map = RandomFourierFeatures(bottleneck_dim, nrf, gamma)
        self.feature_map.new_feature_map()

    def forward(self, x):
        x = self.feature_map(x)
        if self.type == "bn":
            x = self.bn2(x)
        return x

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en
    
def combine(data, taskcode_i):
    bs = data.shape[0]
    data = data.view(bs, -1).cuda()
    taskcode = taskcode_i.repeat(bs, 1).cuda()
    # print(data.shape, taskcode.shape)
    data2 = torch.cat((data, taskcode), dim=1)
    data2 = data2.view(-1, 4, 32, 32)
    return data2.detach().clone()


def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    
    try:
    # print(out_open)
        label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
        # print(out_open.shape)
    except:
        # print(out_open.shape)
        # print(out_open.size(0), out_open.size(2))
        label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
        # print(label_p.shape)
    
    label_range = torch.arange(0, out_open.size(0)).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.sum(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)) ##maxx
    return open_loss_pos, open_loss_neg

def ova_pos_loss(model, extra_components, mem_data): #just for finetuning old tasks
    "contraints for data from the same task" 
    loss = 0
    n = 0
    (model_add, mlp, taskcode, ood_heads) = extra_components
    model.eval()
    model_add.train()
    ood_heads.train()
    mlp.eval()
    for t, data in mem_data.items():
        x_t, y_t = data
        _, fea = model(x_t.cuda(), get_latent=True)
        embed_1  = mlp(taskcode[-1])
        data_add = combine(x_t, embed_1)
        output_add, fea_add = model_add(data_add.detach().clone(), get_latent=True)            
        fea = fea_combined = fea.detach() + fea_add

        out_ood = ood_heads(fea, t)
        out_ood = out_ood.view(x_t.size(0), 2, -1)
        out_ood = F.softmax(out_ood, 1)
        
        label_p = torch.zeros((out_ood.size(0),
                           out_ood.size(2))).long().cuda()
        label_range = torch.range(0, out_ood.size(0) - 1).long()
        label_p[label_range, y_t] = 1
        label_n = 1 - label_p
    
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_ood[:, 1, :]
                                                    + 1e-8) * label_p, 1))
        
        open_loss_neg = torch.mean(torch.sum(-torch.log(out_ood[:, 0, :] +
                                                1e-8) * label_n, 1)) ##maxx
        
        loss += open_loss_pos + open_loss_neg
        n += 1

    return loss/n if n > 0 else 0

def ova_neg_loss(model, extra_components, mem_data, curr_task, new_task=False):  
    "contraints for data from the other tasks"  
    loss = 0
    n = 0
    label_n = 1
    (model_add, mlp, taskcode, ood_heads) = extra_components
    model.eval()
    model_add.train()
    ood_heads.train()
    mlp.eval()
    if new_task: #lean ood for current task
        for t, data in mem_data.items():
            x_t, y_t = data
            _, fea_t = model(x_t.cuda(), get_latent=True)
            embed_1  = mlp(taskcode[-1])
            data_add = combine(x_t, embed_1)
            output_add, fea_add = model_add(data_add.detach().clone(), get_latent=True)            
            fea_t = fea_combined = fea_t.detach() + fea_add
            
            out_ood_t = ood_heads(fea_t, curr_task)
            out_ood_t = out_ood_t.view(x_t.size(0), 2, -1)
            out_ood_t = F.softmax(out_ood_t, 1)
            open_loss_neg_t = torch.mean(torch.sum(-torch.log(out_ood_t[:, 0, :] +
                                                1e-8) * label_n, 1))###
            loss += open_loss_neg_t
            n += 1
            
    else: #finetune for heads of old tasks
        for t, data in mem_data.items():

            x_t, y_t = data
            _, fea = model(x_t.cuda(), get_latent=True)
            embed_1  = mlp(taskcode[-1])
            data_add = combine(x_t, embed_1)
            output_add, fea_add = model_add(data_add.detach().clone(), get_latent=True)            
            fea = fea_combined = fea.detach() + fea_add
            
            for task in range(curr_task + 1):
                if task == t: 
                    continue
                out_ood = ood_heads(fea, task)
                out_ood = out_ood.view(x_t.size(0), 2, -1)
                out_ood = F.softmax(out_ood, 1)
                open_loss_neg = torch.mean(torch.sum(-torch.log(out_ood[:, 0, :] +
                                                1e-8) * label_n, 1))###
                loss += open_loss_neg
                n += 1
                
    loss = loss/n if n > 0 else 0
    
    return loss

def open_finetune(model, extra_components, mem_data, curr_task):
    return 0.5* (ova_neg_loss(model, extra_components, mem_data, curr_task) + ova_pos_loss(model, extra_components, mem_data)) 


def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open