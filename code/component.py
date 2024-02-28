import torch
import torch.nn as nn

def criterion(outputs, targets):
    return nn.MSELoss()(outputs.view(-1), targets.view(-1))


class MultiSampleClassifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(MultiSampleClassifier, self).__init__()
        self.args = args

        self.linear = nn.Linear(input_dim, num_labels)

        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(args.dropout_rate) for _ in range(self.args.dropout_num)]
        )

    def forward(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)

            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits

        if self.args.ms_average:
            logits = logits / self.args.dropout_num

        return logits


class WeightedLayerPooling(nn.Module):
    def __init__(self,num_hidden_layer,layer_start:int = 4,layer_weights = None):
        super(WeightedLayerPooling).__init__()
        self.layer_start = layer_start
        self.num_hidden_layer = num_hidden_layer
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layer+1 - layer_start), dtype=torch.float)
        )

    def forward(self,ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:,:,:,:]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weight_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weight_average


class AttentionPooling(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim,1)
        )

    def forward(self,last_hidden_sate,attention_mask):
        w = self.attention(last_hidden_sate).float()
        w[attention_mask==0] = float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_sate,dim=1)
        return attention_embeddings


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling,self).__init__()

    def forward(self,last_hidden_sate,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_sate.size())
        sum_embeddings = torch.sum(last_hidden_sate * input_mask_expanded,1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask,min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return mean_embedding

