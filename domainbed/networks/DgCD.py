
import torch
import torch.nn as nn
import torch.nn.functional as F

class Scaling(nn.Module):
    def __init__(self, batch_size, channels):
        super(Scaling, self).__init__()
        self.mean = nn.Parameter(torch.zeros((batch_size,channels),device="cuda")) # \beta
        self.var =  nn.Parameter(torch.ones((batch_size,channels),device="cuda")) # \gamma
    
    def forward(self, x):
        return x * self.var + self.mean

class DgCD(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        
    def forward(self, x, ratio, rho):
        batch_size, channels, h, w = x.shape
        avg = self.pool(x.detach().cpu()).view(batch_size,-1).cuda() # global average pooling on Z to get V.
        avg.requires_grad_(True)
        
        '''
        scaler_t = \gamma^t, \beta^t
        scaler_e = \gamma^k, \beta^k
        '''
        scaler_t = Scaling(batch_size,channels)
        batch_env = [[i,i+32] for i in range(0,batch_size,32)]
        scaler_e = []
        for i in range(len(batch_env)):
            scaler_e.append(Scaling(32,channels))
        
        env_wise_aligned, total_alinged = self.alignment(avg, scaler_t, scaler_e)
        total_alinged = F.log_softmax(total_alinged, dim=1)
        env_wise_aligned = F.log_softmax(env_wise_aligned, dim=1)
        dist = self.kl_loss(total_alinged,env_wise_aligned) # Eq. (6)
        dist.backward()
        for p in scaler_t.parameters():
            grad = p.grad
            eps = rho * grad.div(torch.sqrt(torch.norm(grad,dim=-1,keepdim=True) + 1e-12))
            p.data = p.data + eps # Eq. (7)
        for s in scaler_e:
            for p in s.parameters():
                grad = p.grad
                eps = rho * grad.div(torch.sqrt(torch.norm(grad,dim=-1,keepdim=True) + 1e-12))
                p.data = p.data + eps # Eq. (8)
        avg.requires_grad_(False)
        
        env_wise_aligned, total_alinged = self.alignment(avg, scaler_t, scaler_e) # Eq. (10)
        
        gram = torch.mm(total_alinged.t()+ 1e-7,env_wise_aligned + 1e-7).diag() # Gram matrix in Eq. (11)
        total_alinged = self.scaling(total_alinged/gram) 
        env_wise_aligned = self.scaling(env_wise_aligned/gram)
        scores = (total_alinged - env_wise_aligned).pow(2) # Eq. (11)
        
        scores = self.scaling(scores)
        mask_filters = self.make_mask(scores,ratio) # Eq. (12-13)
        return x * mask_filters # Eq. (14)
    
    def alignment(self, x, scaler_t, scaler_e):
        batch_size = len(x)
        batch_env = [[i,i+32] for i in range(0,batch_size,32)]
        env_wise_aligned = []
        for i in range(len(batch_env)):
            sample = x[batch_env[i][0]:batch_env[i][1]].clone()
            s_m = sample.mean(dim=0, keepdim=True)
            s_v = sample.var(dim=0, keepdim=True)
            sample = (sample - s_m) / torch.sqrt(s_v + 1e-5) # Eq. (4)
            sample = scaler_e[i](sample) # Eq. (5)
            env_wise_aligned.append(sample)
        env_wise_aligned = torch.cat(env_wise_aligned,dim=0) # Eq. (5)

        t_m = x.mean(dim=0, keepdim=True)
        t_v = x.var(dim=0, keepdim=True)
        total_alinged = (x - t_m) / torch.sqrt(t_v + 1e-5) # Eq. (4)
        total_alinged = scaler_t(total_alinged) # Eq. (5)
        return env_wise_aligned, total_alinged
    
    def scaling(self,x): # scaler in Eq. (11-12)
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_min = x.min(dim=-1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min)
        return x
        
    def make_mask(self,scores,ratio):
        batch_size, channels = scores.shape
        r = torch.rand(scores.shape).cuda()  # BxC
        key1 = r.pow(1. / scores) # Eq. (12)
        threshold = torch.sort(key1, dim=1, descending=True)[0][:, int(ratio*channels)]
        threshold_expand = threshold.view(batch_size, 1).expand(batch_size, channels)
        mask_filters = torch.where(key1 > threshold_expand, torch.tensor(1.).cuda(), torch.tensor(0.).cuda()) # Eq. (13)
        mask_filters = 1 - mask_filters
        mask_filters = mask_filters.view(batch_size, channels, 1, 1)
        mask_filters = mask_filters * mask_filters.numel() / mask_filters.sum()
        return mask_filters

