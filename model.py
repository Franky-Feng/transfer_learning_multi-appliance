import math
import torch
from torch import nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT4NILM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.original_len = args.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = args.drop_out
        self.hidden = 256
        self.heads = 2
        self.n_layers = 2
        self.output_size = args.output_size
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
                              padding_mode='replicate')
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)  # 240 256
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in
             range(self.n_layers)])
        self.deconv = nn.ConvTranspose1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2,
                                         padding=1)
        # self.linear1 = nn.Linear(self.hidden, 128)
        # self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        # print(params)
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, sequence):
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1))
        # x = torch.tanh(self.linear1(x))
        # x = self.linear2(x)
        return x



class CombinedModel(nn.Module):
    def __init__(self, pretrained_model, new_model, new_model2, new_model3, new_model4, new_model5):
        super(CombinedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.new_model = new_model
        self.new_model2 = new_model2
        self.new_model3 = new_model3
        self.new_model4 = new_model4
        self.new_model5 = new_model5

    def forward(self, sequence):
        x = self.pretrained_model(sequence)
        # x = x.squeeze()
        # x = F.relu(x)
        y, s = self.new_model(x)
        y2, s2 = self.new_model2(x)
        y3, s3 = self.new_model3(x)
        y4, s4 = self.new_model4(x)
        y5, s5 = self.new_model5(x)
        combined_tensor = torch.cat([y, y2, y3, y4, y5], dim=2)
        combined_tensor_status = torch.cat([s, s2, s3, s4, s5], dim=2)
        return combined_tensor, combined_tensor_status


class CombinedModel3(nn.Module):
    def __init__(self, pretrained_model, new_model, new_model2, new_model3, new_model4):
        super(CombinedModel3, self).__init__()
        self.pretrained_model = pretrained_model
        self.new_model = new_model
        self.new_model2 = new_model2
        self.new_model3 = new_model3
        self.new_model4 = new_model4

    def forward(self, sequence):
        x = self.pretrained_model(sequence)
        # x = x.squeeze()
        # x = F.relu(x)
        y, s = self.new_model(x)
        y2, s2 = self.new_model2(x)
        y3, s3 = self.new_model3(x)
        y4, s4 = self.new_model4(x)
        combined_tensor = torch.cat([y, y2, y3, y4], dim=2)
        combined_tensor_status = torch.cat([s, s2, s3, s4], dim=2)
        return combined_tensor, combined_tensor_status


class CombinedModel2(nn.Module):
    def __init__(self, pretrained_model, new_model1):
        super(CombinedModel2, self).__init__()
        self.pretrained_model = pretrained_model
        self.new_model1 = new_model1

    def forward(self, sequence):
        x = self.pretrained_model(sequence)
        # x = x.squeeze()
        # x = F.relu(x)
        y, s = self.new_model1(x)
        y = y.squeeze()
        s = s.squeeze()
        # combined_tensor = torch.cat([y], dim=2)
        # combined_tensor_status = torch.cat([s], dim=2)
        return y, s


class CNNNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout_rate = args.drop_out
        self.original_len = args.window_size
        self.hidden = 256
        self.heads = 2

        self.conv = nn.Conv1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
                              padding_mode='replicate')
        # self.conv2 = nn.Conv1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
        #                        padding_mode='replicate')
        # self.conv3 = nn.Conv1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
        #                        padding_mode='replicate')
        # self.conv4 = nn.Conv1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
        #                        padding_mode='replicate')
        # self.conv5 = nn.Conv1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
        #                        padding_mode='replicate')
        self.layer_norm = LayerNorm(self.hidden)
        self.layer_norm2 = LayerNorm(self.hidden)

        # self.layer_norm1 = LayerNorm(self.original_len)

        # self.batch_norm_layer = nn.BatchNorm1d(480)
        # self.flatten = nn.Linear(self.original_len * self.hidden, self.original_len)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, 1)
        # self.position = PositionalEmbedding(max_len=self.original_len, d_model=self.hidden)  # 480 256
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in
             range(1)])
        self.fc = nn.Sequential(nn.Linear(self.hidden, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1))

        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        # print(params)
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, sequence):
        # mean = torch.mean(sequence, dim=1, keepdim=True)
        # std = torch.std(sequence, dim=1, keepdim=True)
        # normalized_tensor = (sequence - mean) / std
        # x = self.layer_norm1(sequence)
        x = self.conv(sequence).permute(0, 2, 1)

        # embedding = x + self.position(sequence)
        x = self.dropout(self.layer_norm(x))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x = self.dropout2(self.layer_norm2(x))

        s = torch.sigmoid(self.fc(x))
        y = torch.tanh(self.linear1(x))
        y = self.linear2(y) * s

        # x = self.flatten(x)
        # x = x.unsqueeze(2)
        return y, s


class shared_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.original_len = args.window_size  # 480
        self.latent_len = int(self.original_len / 2)  # 240
        self.dropout_rate = args.drop_out  # dropout_rate=0.1

        self.hidden = 256
        self.heads = 2
        self.n_layers = 2
        self.output_size = args.output_size

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden, kernel_size=5, stride=1, padding=2,
                              padding_mode='replicate')
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)  # 240 256
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in
             range(self.n_layers)])

        self.deconv = nn.ConvTranspose1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2,
                                         padding=1)
        # self.linear1 = nn.Linear(self.hidden, 128)
        # self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        # print(params)
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, sequence):
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))
        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1))
        # x = torch.tanh(self.linear1(x))
        # x = self.linear2(x)
        return x
