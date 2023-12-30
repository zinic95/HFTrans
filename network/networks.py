# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm, MaxPool3d
from torch.nn.modules.utils import _triple
from .configs import *
from torch.distributions.normal import Normal


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

class Attention_m(nn.Module):
    def __init__(self, config, vis):
        super(Attention_m, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size()[-2]*self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        n_patches = hidden_states.size()[1]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (n_patches, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.transpose(-2,-3)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Attention_ch(nn.Module):
    def __init__(self, config, vis):
        super(Attention_ch, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-2] + (x.size()[-2]*self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (5, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class CrossAttention(nn.Module):
    def __init__(self, config, hidden_size):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def forward(self, hidden_states_q, hidden_states_kv):
        # hidden_states = (B, C, W, H, D)
        size = hidden_states_kv.size()
        hidden_states_q = hidden_states_q.flatten(2,4)
        hidden_states_q = hidden_states_q.permute(0,2,1)
        hidden_states_kv = hidden_states_kv.flatten(2,4)
        hidden_states_kv = hidden_states_kv.permute(0,2,1)

        query_layer = self.query(hidden_states_q)
        key_layer = self.key(hidden_states_kv)
        value_layer = self.value(hidden_states_kv)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        attention_output = attention_output.permute(0,2,1)
        attention_output = attention_output.view(size)
        return attention_output

class ChannelAttention(nn.Module):
    def __init__(self, config, hidden_size):
        super(ChannelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.maxpool = MaxPool3d(2, stride=2)

    def forward(self, hidden_states):
        # hidden_states = (B*M, C, S)
        size = hidden_states.size()

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_scores = torch.sum(attention_scores, (1))
        
        attention_probs = self.softmax(attention_scores) # B*M x C
        attention_probs = self.attn_dropout(attention_probs)
        return attention_probs

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings_HFTrans_middle(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTrans_middle, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = num_encoders * int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        
        #self.early_encoder =CNNEncoder_stride(config, input_channels)
        hybrid_encoders = []
        for i in range(num_encoders):
            hybrid_encoders.append(CNNEncoder_stride(config, 1))
        self.hybrid_encoders = nn.ModuleList(hybrid_encoders)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches // num_encoders, num_encoders, config.hidden_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.num_encoders = num_encoders

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = []
        features_enc = []
        for i, hybrid_encoder in enumerate(self.hybrid_encoders):
            xx, features = hybrid_encoder(x[:,i:i+1,:,:,:])
            x_enc.append(xx)
            if features_enc == []:
                features_enc = [[] for f in features]
            for n, f in enumerate(features):
                features_enc[n].append(f)

        x_enc = [self.patch_embeddings(x) for x in x_enc]
        x_enc = [torch.unsqueeze(x, 5) for x in x_enc]
        
        x = torch.cat(x_enc, 5)
        x = x.flatten(2,4) 
        x = x.permute(0,2,3,1)

        features = [torch.cat((f), 1) for f in features_enc]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Embeddings_HFTransb2s(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTransb2s, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        
        self.early_encoder =CNNEncoder_stride(config, input_channels)

        in_channels = config['encoder_channels'][-1]
        self.num_modality = num_encoders-1
        self.patch_embeddings = Conv3d(in_channels=in_channels // self.num_modality,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.num_modality = num_encoders-1
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.num_modality, config.hidden_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.sc_trans = SCTrans(self.config)
        sc_cnn = []
        for ch in config.skip_channels:
            sc_cnn.append(Conv3d(in_channels=ch*self.num_modality,
                                out_channels=ch,
                                kernel_size=1,
                                stride=1))
        self.sc_cnn = nn.ModuleList(sc_cnn)

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x, features_enc = self.early_encoder(x) # all
        B, C, W, H, D = x.size() # B x C x W x H x D
        x = x.reshape(B * self.num_modality, C // self.num_modality, H, W, D)
        x = self.patch_embeddings(x)
        _, C, W, H, D = x.size()
        x = x.reshape(B, self.num_modality, C, W, H, D)

        x = x.flatten(3,5) 
        x = x.permute(0,3,1,2)

        features = []
        for i, f in enumerate(features_enc):
            features.append(self.sc_cnn[i](f))
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Embeddings_HFTrans(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTrans, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = num_encoders * int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        
        self.early_encoder =CNNEncoder_stride(config, input_channels)
        hybrid_encoders = []
        for i in range(num_encoders-1):
            hybrid_encoders.append(CNNEncoder_stride(config, 1))
        self.hybrid_encoders = nn.ModuleList(hybrid_encoders)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches // num_encoders, num_encoders, config.hidden_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.num_encoders = num_encoders

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = []
        xx, features = self.early_encoder(x) # all
    
        x_enc.append(xx)
        features_enc = [[f] for f in features]
        for i, hybrid_encoder in enumerate(self.hybrid_encoders):
            xx, features = hybrid_encoder(x[:,i:i+1,:,:,:])
            x_enc.append(xx)
            for n, f in enumerate(features):
                features_enc[n].append(f)

        x_enc = [self.patch_embeddings(x) for x in x_enc]
        x_enc = [torch.unsqueeze(x, 5) for x in x_enc]
        
        x = torch.cat(x_enc, 5)
        x = x.flatten(2,4) 
        x = x.permute(0,2,3,1)

        features = [torch.cat((f), 1) for f in features_enc]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Embeddings_HFTransSimple(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTransSimple, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = num_encoders * int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        
        self.early_encoder =CNNEncoder_stride(config, input_channels)
        hybrid_encoders = []
        for i in range(num_encoders-1):
            hybrid_encoders.append(CNNEncoder_stride(config, 1))
        self.hybrid_encoders = nn.ModuleList(hybrid_encoders)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches // num_encoders, num_encoders, config.hidden_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.num_encoders = num_encoders

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = []
        xx, features = self.early_encoder(x) # all
    
        x_enc.append(xx)
        for i, hybrid_encoder in enumerate(self.hybrid_encoders):
            xx, f = hybrid_encoder(x[:,i:i+1,:,:,:])
            x_enc.append(xx)
            features = [features[i] + f[i] for i in range(len(features))]

        x_enc = [self.patch_embeddings(x) for x in x_enc]
        x_enc = [torch.unsqueeze(x, 5) for x in x_enc]
        
        x = torch.cat(x_enc, 5)
        x = x.flatten(2,4) 
        x = x.permute(0,2,3,1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class SCTrans(nn.Module):
    def __init__(self, config):
        super(SCTrans, self).__init__()
        self.config = config
        self.maxpool = MaxPool3d(2, stride=2)
        self.linear = Linear(512,512)
        self.att = ChannelAttention(config, 512)
    def forward(self, x):
        x = torch.cat((x), 5)
        B, C, W, H, D, M = x.size() # B x C x W x H x D x M
        x = x.permute(0,5,1,2,3,4)
        x = x.flatten(0,1) # B*M x C x W x H x D
        y = x  
        while y.size()[2] != 8:
            y = self.maxpool(y)
        # B*M x C x 8 x 8 x 8
        y = y.flatten(2,4) # B*M x C x 512
        y = self.linear(y)
        attention_probs = self.att(y) # B*M x C
        x = x * attention_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(B, M, C, W, H, D) # B x M x C x 512
        x = torch.sum(x, (1)) # B x C x W x H x D
        return x 

class Embeddings_HFTransSC(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTransSC, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = num_encoders * int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        
        self.early_encoder =CNNEncoder_stride(config, input_channels)
        hybrid_encoders = []
        for i in range(num_encoders-1):
            hybrid_encoders.append(CNNEncoder_stride(config, 1))
        self.hybrid_encoders = nn.ModuleList(hybrid_encoders)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches // num_encoders, num_encoders, config.hidden_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.num_encoders = num_encoders
        self.sc_trans = SCTrans(self.config)
        sc_cnn = []
        for ch in config.skip_channels:
            sc_cnn.append(Conv3d(in_channels=ch*num_encoders,
                                out_channels=ch,
                                kernel_size=1,
                                stride=1))
        self.sc_cnn = nn.ModuleList(sc_cnn)

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = []
        xx, features = self.early_encoder(x) # all
    
        x_enc.append(xx)
        #features_enc = [[torch.unsqueeze(f, 5)] for f in features]
        features_enc = [[f] for f in features]
        for i, hybrid_encoder in enumerate(self.hybrid_encoders):
            xx, features = hybrid_encoder(x[:,i:i+1,:,:,:])
            x_enc.append(xx)
            for n, f in enumerate(features):
                features_enc[n].append(f)

        x_enc = [self.patch_embeddings(x) for x in x_enc]
        x_enc = [torch.unsqueeze(x, 5) for x in x_enc]
        
        x = torch.cat(x_enc, 5)
        x = x.flatten(2,4) 
        x = x.permute(0,2,3,1)

        features = []
        for i, f in enumerate(features_enc):
            features.append(self.sc_cnn[i](torch.cat((f), 1)))
            #features.append(self.sc_trans(f))
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Embeddings_HFTransCA(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTransCA, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = num_encoders * int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        in_channels = config['encoder_channels'][-1]
        self.early_encoder =CNNEncoder_stride(config, input_channels)
        hybrid_encoders = []
        CAs = [] 
        for i in range(num_encoders-1):
            hybrid_encoders.append(CNNEncoder_stride(config, 1))
            CAs.append(CrossAttention(config,in_channels))
        self.hybrid_encoders = nn.ModuleList(hybrid_encoders)
        self.CAs = nn.ModuleList(CAs)
        
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches // num_encoders, num_encoders, config.hidden_size))
        
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.num_encoders = num_encoders

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = []
        xx, features = self.early_encoder(x) # all
    
        x_enc.append(xx)
        features_enc = [[f] for f in features]
        for i, hybrid_encoder in enumerate(self.hybrid_encoders):
            xx, features = hybrid_encoder(x[:,i:i+1,:,:,:])
            xx = self.CAs[i](x_enc[0], xx)
            x_enc.append(xx)
            for n, f in enumerate(features):
                features_enc[n].append(f)

        x_enc = [self.patch_embeddings(x) for x in x_enc]
        x_enc = [torch.unsqueeze(x, 5) for x in x_enc]
        
        x = torch.cat(x_enc, 5)
        x = x.flatten(2,4) 
        x = x.permute(0,2,3,1)

        features = [torch.cat((f), 1) for f in features_enc]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer_HFTrans(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTrans, self).__init__()
        self.embeddings = Embeddings_HFTrans(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class Transformer_HFTransb2s(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTransb2s, self).__init__()
        self.embeddings = Embeddings_HFTransb2s(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class Transformer_HFTransSimple(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTransSimple, self).__init__()
        self.embeddings = Embeddings_HFTransSimple(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class Transformer_HFTransCA(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTransCA, self).__init__()
        self.embeddings = Embeddings_HFTransCA(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class Transformer_HFTransSC(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTransSC, self).__init__()
        self.embeddings = Embeddings_HFTransSC(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class Transformer_HFTrans_middle(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTrans_middle, self).__init__()
        self.embeddings = Embeddings_HFTrans_middle(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class ConsecutiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = 0):
        super(ConsecutiveConv, self).__init__()

        if mid_channels == 0:
            mid_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.convs(x)

class ConsecutiveConv_res(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = 0):
        super(ConsecutiveConv_res, self).__init__()

        if mid_channels == 0:
            mid_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.convs(x) + x

class CNNEncoderBlock_stride(nn.Module):
    """Downscaling with strided convolution then max pooling"""

    def __init__(self, in_channels, out_channels, stride):
        super(CNNEncoderBlock_stride, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, stride=stride),
            ConsecutiveConv_res(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class CNNEncoder_stride(nn.Module):
    def __init__(self, config, n_channels=1):
        super(CNNEncoder_stride, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = ConsecutiveConv(n_channels, encoder_channels[0])

        blocks = [
            CNNEncoderBlock_stride(encoder_channels[i], encoder_channels[i+1], config.down_factor) for i in range(self.down_num)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        x = self.inc(x)
        features.append(x)
        for encoder_block in self.blocks:
            x = encoder_block(x)
            features.append(x)

        return x, features[::-1][1:]

class ConsecutiveConv_up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(ConsecutiveConv_up, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x, feat):
        x = self.conv1(x)
        x = self.conv2(x)
        if feat is not None:
            x = torch.cat((x,feat), dim=1)
        x = self.conv3(x)
        return x

class CNNDecoderBlock_transpose(nn.Module):
    """Upsampling with transposed convolution"""

    def __init__(self, in_channels, out_channels, skip_channels):
        super(CNNDecoderBlock_transpose, self).__init__()
        self.upblock = ConsecutiveConv_up(in_channels, out_channels, skip_channels)
        self.block = ConsecutiveConv_res(out_channels, out_channels)

    def forward(self, x, feat):
        x = self.upblock(x, feat)
        x = self.block(x)
        return x

class DecoderHFTrans(nn.Module):
    def __init__(self, config, img_size, num_encoders):
        super().__init__()
        self.config = config
        self.down_num = config.down_num
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = ConsecutiveConv(config.hidden_size*num_encoders, head_channels)

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            CNNDecoderBlock_transpose(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        if max(self.patch_size) != 1:
            self.up = nn.Upsample(scale_factor=self.patch_size, mode='trilinear', align_corners=False)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size() 
        l, h, w = (self.img_size[0]//2**self.down_num//self.patch_size[0]), (self.img_size[1]//2**self.down_num//self.patch_size[1]), (self.img_size[2]//2**self.down_num//self.patch_size[2])
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        if max(self.patch_size) != 1:
            x = self.up(x)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class HFTrans(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, num_encoders=5, vis=False):
        super(HFTrans, self).__init__()
        self.transformer = Transformer_HFTrans(config, img_size, vis, input_channels, num_encoders)
        self.decoder = DecoderHFTrans(config, img_size, num_encoders)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.num_encoders = num_encoders
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_encoders, self.num_encoders*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights

class HFTransb2s(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, num_encoders=5, vis=False):
        super(HFTransb2s, self).__init__()
        self.transformer = Transformer_HFTransb2s(config, img_size, vis, input_channels, num_encoders)
        self.num_modality = num_encoders-1
        self.decoder = DecoderHFTrans(config, img_size, self.num_modality)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_modality, self.num_modality*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights

class HFTransSimple(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, num_encoders=5, vis=False):
        super(HFTransSimple, self).__init__()
        self.transformer = Transformer_HFTransSimple(config, img_size, vis, input_channels, num_encoders)
        self.decoder = DecoderHFTrans(config, img_size, num_encoders)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.num_encoders = num_encoders
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_encoders, self.num_encoders*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights

class HFTransSC(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, num_encoders=5, vis=False):
        super(HFTransSC, self).__init__()
        self.transformer = Transformer_HFTransSC(config, img_size, vis, input_channels, num_encoders)
        self.decoder = DecoderHFTrans(config, img_size, num_encoders)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.num_encoders = num_encoders
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_encoders, self.num_encoders*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights


class HFTransCA(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, num_encoders=5, vis=False):
        super(HFTransCA, self).__init__()
        self.transformer = Transformer_HFTransCA(config, img_size, vis, input_channels, num_encoders)
        self.decoder = DecoderHFTrans(config, img_size, num_encoders)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.num_encoders = num_encoders
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_encoders, self.num_encoders*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights


class HFTrans_middle(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, num_encoders=5, vis=False):
        super(HFTrans_middle, self).__init__()
        self.transformer = Transformer_HFTrans_middle(config, img_size, vis, input_channels, num_encoders)
        self.decoder = DecoderHFTrans(config, img_size, num_encoders)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.num_encoders = num_encoders
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_encoders, self.num_encoders*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights

CONFIGS = {
    'HFTrans_16' : get_HFTrans_16_config(),
    'HFTrans_64' : get_HFTrans_64_config(),
    'HFTrans_16b2s' : get_HFTrans_16b2s_config(),
    'HFTrans5_16' : get_HFTrans5_16_config(),
    'HFTrans5_32' : get_HFTrans5_32_config(),
    'HFTrans4_16' : get_HFTrans4_16_config(),
    'HFTrans4_32' : get_HFTrans4_32_config(),
}