import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wide(nn.Module):
    def __init__(self,input_dim):
        super(Wide,self).__init__()
        self.linear = nn.Linear(in_features=input_dim,out_features=1)

    def forward(self,x):
        return self.linear(x)

class Deep(nn.Module):
    def __init__(self,hidden_units,dropout=0.):
        super(Deep,self).__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(layer[0],layer[1]) for layer in list(zip(hidden_units[:-1],hidden_units[1:]))]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.dropout(x)

        return x

class Model(nn.Module):
    def __init__(self,feature_columns,dropout=0.,gpu=False):
        super(Model,self).__init__()
        self.gpu = gpu

        self.dense_feature_cols,self.sparse_feature_cols = feature_columns
        self.embed_layers = nn.ModuleDict({
            'embed_'+str(i):nn.Embedding(num_embeddings=feat['feat_num'],embedding_dim=feat['embed_dim'])
            for i,feat in enumerate(self.sparse_feature_cols)
        })

        # client has DeepBottom1 and category features
        hidden1 = [len(self.sparse_feature_cols) * self.sparse_feature_cols[0]['embed_dim'],192,96,48]
        self.bottom1 = Deep(hidden1)
        # server has DeepBottom2 + DeepTop + Wide and real-value features
        hidden2 = [len(self.dense_feature_cols),64,32,16]
        self.bottom2 = Deep(hidden2)
        hidden3 = [64,32]
        self.top = Deep(hidden3,dropout)
        self.final_layer = nn.Linear(hidden3[-1],1)
        self.wide = Wide(input_dim=len(self.dense_feature_cols))

        if gpu:
            self.cuda()

    def forward(self,x):
        if self.gpu:
            x = x.cuda()

        dense_inputs,sparse_inputs = x[:,:len(self.dense_feature_cols)], x[:,len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_'+str(i)](sparse_inputs[:,i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds,axis=-1)

        wide_out = self.wide(dense_inputs)
        deep_out = self.top(torch.cat([self.bottom1(sparse_embeds),self.bottom2(dense_inputs)],axis=-1))
        deep_out = self.final_layer(deep_out)

        out = torch.sigmoid(0.5*(wide_out+deep_out))

        return out
