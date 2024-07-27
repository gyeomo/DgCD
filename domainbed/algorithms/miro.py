# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from domainbed.optimizers import get_optimizer
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from domainbed.algorithms import Algorithm


class ForwardModel(nn.Module):
    """Forward model is used to reduce gpu memory usage of SWAD.
    """
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        return self.network(x)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, int(hidden_size)),
                                    nn.ReLU(),
                                    nn.Linear(int(hidden_size), 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class TCLineEstimator(nn.Module):
    def __init__(self, dims, hidden_size=None, mi_estimator='CLUB'):  
        super().__init__()
        self.dims = dims
        MI_CLASS={
            'CLUB': MINE,
            }
        self.mi_est_type = MI_CLASS[mi_estimator]

        mi_estimator_list = [
            self.mi_est_type(
                x_dim=sum(dims[:i+1]),
                y_dim=dim,
                hidden_size=(None if hidden_size is None else hidden_size * np.sqrt(i+1))
            )
            for i, dim in enumerate(dims[:-1])
        ]
        self.mi_estimators = nn.ModuleList(mi_estimator_list)
    
    def forward(self, samples):
        outputs = []
        concat_samples = [samples[0]]
        for i, dim in enumerate(self.dims[1:]):
            cat_sample = torch.cat(concat_samples, dim=1)
            outputs.append(self.mi_estimators[i](cat_sample, samples[i+1]))
            concat_samples.append(samples[i+1])
        return torch.stack(outputs).sum()

    def learning_loss(self, samples):
        outputs = []
        concat_samples = [samples[0]]
        for i, dim in enumerate(self.dims[1:]):
            cat_sample = torch.cat(concat_samples, dim=1)
            outputs.append(self.mi_estimators[i].learning_loss(cat_sample, samples[i+1]).mean())
            concat_samples.append(samples[i+1])
        return torch.stack(outputs).mean()


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]

    return shapes


class MIRO(Algorithm):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld

        # build mean/var encoders
        shapes = get_shapes(self.featurizer, self.input_shape)
        self.class_mi = nn.ModuleList([
                                 TCLineEstimator(
                                 dims=[32 for _ in range(3)], 
                                 hidden_size=128, 
                                 mi_estimator="CLUB") for shape in shapes
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        # optimizer
        parameters = [
            {"params": self.network.parameters()},
            {"params": self.class_mi.parameters(), "lr": hparams.lr * hparams.lr_mult},
            #{"params": self.var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        batch_size, channels, h, w = all_x.shape
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)
        
        reg_loss = 0.
        for n,f in enumerate(inter_feats):
            f = self.pool(f).view(batch_size,-1)
            batch_env = [[i,i+32] for i in range(0,batch_size,32)]
            env_batch = []
            for i in range(len(batch_env)):
                env_batch.append(f[batch_env[i][0]:batch_env[i][1]])
            f = torch.stack(env_batch).permute(0,2,1)
            reg_loss += self.class_mi[n].learning_loss(f)

        loss += reg_loss * self.ld
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "reg": reg_loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model
