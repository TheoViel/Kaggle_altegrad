from params import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import *


TRANSFORMERS = {
    "bert-base-uncased": (BertModel, BertTokenizer, "bert-base-uncased"),
    "bert-base-multilingual-uncased": (BertModel, BertTokenizer, "bert-base-multilingual-uncased",),
    "camembert-base": (CamembertModel, CamembertTokenizer, "camembert-base"),
}


class BertMultiPooler(nn.Module):
    """
    Custom Bert Pooling head that takes the n last layers
    """

    def __init__(self, nb_layers=1, input_size=768, nb_ft=768):
        """
        Constructor
        
        Arguments:
            nb_layers {int} -- Number of layers to consider (default: {1})
            input_size {int} -- Size of the input features (default: {768})
            nb_ft {int} -- Size of the output features (default: {768})
        """
        super().__init__()

        self.nb_layers = nb_layers
        self.input_size = input_size
        self.poolers = nn.ModuleList([])

        for i in range(nb_layers):
            pooler = nn.Sequential(nn.Linear(input_size, nb_ft), nn.Tanh(),)
            self.poolers.append(pooler)

    def forward(self, hidden_states, idx=0):
        """
        Usual torch forward function
        
        Arguments:
            hidden_states {list of torch tensors} -- Hidden states of the model, the last one being at index 0
        
        Keyword Arguments:
            idx {int} -- Index to apply the pooler on (default: {0})
        
        Returns:
            torch tensor -- Pooled features
        """
        bs = hidden_states[0].size()[0]
        if type(idx) == int:
            idx = torch.tensor([idx] * bs).cuda()

        outputs = []
        idx = idx.view(-1, 1, 1).repeat(1, 1, self.input_size)

        for i, (state) in enumerate(hidden_states[: self.nb_layers]):
            token_tensor = state.gather(1, idx).view(bs, -1)

            pooled = self.poolers[i](token_tensor)
            outputs.append(pooled)

        return torch.cat(outputs, -1)


class Transformer(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, avg_pool=False):
        """
        Constructor
        
        Arguments:
            model {string} -- Transformer to build the model on. Expects "camembert-base".
        
        Keyword Arguments:
            nb_layers {int} -- Number of layers to consider for the pooler (default: {1})
            pooler_ft {[type]} -- Number of features for the pooler. If None, use the same number as the transformer (default: {None})
            avg_pool {bool} -- Whether to use average pooling instead of pooling on the first tensor (default: {False})
        """
        super().__init__()
        self.name = model
        self.avg_pool = avg_pool
        self.nb_layers = nb_layers

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )

        self.nb_features = self.transformer.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        if nb_layers != 1:
            self.pooler = BertMultiPooler(
                nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
            )
        else:
            self.pooler = nn.Sequential(
                nn.Linear(self.nb_features, pooler_ft), 
                nn.Tanh(),
            )

        self.logit = nn.Linear(pooler_ft, len(CLASSES))

    def forward(self, tokens):
        """
        Usual torch forward function
        
        Arguments:
            tokens {torch tensor} -- Sentence tokens
        
        Returns:
            torch tensor -- Class logits
            torch tensor -- Pooled features
        """
        _, _, hidden_states = self.transformer(
            tokens, attention_mask=(tokens > 0).long()
        )

        hidden_states = hidden_states[::-1]

        if self.nb_layers == 1:  # Not using the custom pooler
            if self.avg_pool:
                hidden_states = hidden_states[0].mean(1)
            else:
                hidden_states = hidden_states[0][:, 0]

        ft = self.pooler(hidden_states)
        y = self.logit(ft)

        return y, ft
