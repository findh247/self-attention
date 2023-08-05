import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock

# class SelfAttentionPooling(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.W = nn.Linear(input_dim, 1)
#     def forward(self, batch_rep):
#         """
#             input:
#                 batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
#
#             attention_weight:
#                 att_w : size (N, T, 1)
#
#             return:
#                 utter_rep: size (N, H)
#         """
#         att_w = F.softmax(self.W(batch_rep).squeeze(-1), dim=-1).unsqueeze(-1)
#         utter_rep = torch.sum(batch_rep * att_w, dim=1)#序列长度维度进行计算
#         return utter_rep


# medium
class Classifier(nn.Module):
    def __init__(self, d_model=256, n_spks=1251, dropout=0.2):
        super().__init__()
        # 40-->d_model
        self.prenet = nn.Linear(40, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
            dim_feedforward=d_model*2, nhead=2, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
            num_layers=3)
        # self.sap = SelfAttentionPooling(d_model)
        self.pred_layer = nn.Sequential(
          nn.BatchNorm1d(d_model), # 批标准化 减小过拟合
          nn.Linear(d_model, n_spks),  # 输出长为n_spks的概率
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)使得每个语音片段都作为一个单独的序列输入到 Transformer 编码器中，方便进行并行计算。
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # self attention
        stats = out.mean(dim=1)
        # stats = self.sap(out)
        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out