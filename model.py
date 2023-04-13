from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import functional as F
import scipy.sparse as sp
# GCN parameters
GCN_FEATURE_DIM = 1024#gcn_feature_dim
GCN_HIDDEN_DIM = 256 #gcn_hiddem_dim
# GCN_HIDDEN_DIM_2 = 128
GCN_OUTPUT_DIM = 64  #gcn_output_dim

# Attention parameters
DENSE_DIM = 64 #dense_dim=64
ATTENTION_HEADS = 64
NUM_CLASSES = 128

LEARNING_RATE = 1E-4 #learning_rate
WEIGHT_DECAY = 1E-4  #weight_decay

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):      
        support = input @ self.weight    # X * W      
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(GCN_FEATURE_DIM, GCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			# x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        # output = self.relu2(self.ln2(x))	# output.shape = (seq_len, GCN_OUTPUT_DIM)
        output = self.relu2(self.ln2(x))  # output.shape = (seq_len, GCN_OUTPUT_DIM)
        return output


class Attention(nn.Module):

    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  # attention.shape = (1, attention_hops, seq_len)
        return attention
class Textcnn(torch.nn.Module):
    def __init__(self,emb_dim):
        super(Textcnn,self).__init__()
        self.embedding_size = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,out_channels = 128, kernel_size = 3)
        self.mx1 = nn.MaxPool1d(2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=128,out_channels = 128, kernel_size = 3)
        self.mx2 = nn.MaxPool1d(2, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels = 128, kernel_size = 3)
        #液泡
        # self.mx3 = nn.MaxPool1d(432, stride=1)
        #高尔基体
        self.mx3 = nn.MaxPool1d(594, stride=1)
        # 高尔基体_new
        # self.mx3 = nn.MaxPool1d(602, stride=1)
        #线粒体M983
        # self.mx3 = nn.MaxPool1d(387, stride=1)
        #过氧化酶体
        # self.mx3 = nn.MaxPool1d(433, stride=1)
        #线粒体M317
        # self.mx3 = nn.MaxPool1d(383, stride=1)


    def forward(self,x):
        # x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        features = self.conv1(x)
        features = self.mx1(features)
        features = self.mx2(self.conv2(features))
        features = self.conv3(features)
        features = self.mx3(features)
        features = features.squeeze(2)
        return features
# class TextcnnIndepend(torch.nn.Module):
#     def __init__(self,emb_dim):
#         super(TextcnnIndepend,self).__init__()
#         self.embedding_size = emb_dim
#         self.conv1 = nn.Conv1d(in_channels=self.embedding_size,out_channels = 128, kernel_size = 3)
#         self.mx1 = nn.MaxPool1d(2, stride=1)
#         self.conv2 = nn.Conv1d(in_channels=128,out_channels = 128, kernel_size = 3)
#         self.mx2 = nn.MaxPool1d(2, stride=1)
#         self.conv3 = nn.Conv1d(in_channels=128,out_channels = 128, kernel_size = 3)
#         self.mx3 = nn.MaxPool1d(432, stride=1)
#         self.out = nn.Linear(128, 2)
#
#     def forward(self,x):
#         # x = x.squeeze(1)
#         x = x.permute(0, 2, 1)
#         features = self.conv1(x)
#         features = self.mx1(features)
#         features = self.mx2(self.conv2(features))
#         features = self.conv3(features)
#         features = self.mx3(features)
#         features = features.squeeze(2)
#         features=self.out(features)
#         return features
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.output_dim=128
        self.gcn = GCN()
        self.textcnn=Textcnn(1024)
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM * ATTENTION_HEADS, NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        # self.fc1 = nn.Linear(self.output_dim * 2, 512)
        # self.fc1 = nn.Linear(self.output_dim, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
        # self.out = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.drop = 0.5
        self.dropout = nn.Dropout(self.drop)
        self.textflatten = nn.Linear(128, self.output_dim)

    def forward(self, x,adj,x1):
        
        x = x.float()      # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        m=x.shape[0]
        adj = adj.float()
        x = self.gcn(x, adj)  												# x.shape = (seq_len, GAT_OUTPUT_DIM)
        x = x.unsqueeze(0).float()  										# x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x 									# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_con = torch.flatten(node_feature_embedding, start_dim=1) # output.shape = (1, ATTENTION_HEADS * GAT_OUTPUT_DIM)
        # node_feature_embedding_con = torch.flatten(x, start_dim=1)
        # fc_final=nn.Linear(m*GCN_OUTPUT_DIM ,128)
        output1 = self.fc_final(node_feature_embedding_con) 	# output.shape = (1, NUM_CLASSES)
        # output1 = fc_final(node_feature_embedding_con)  # output.shape = (1, NUM_CLASSES)
        g1 = self.relu(output1)
        x1 = x1.unsqueeze(0).float()
        output2 = self.textcnn(x1)
        output2 = self.relu(self.textflatten(output2))
        # output2 = self.textflatten(output2)
        w1 = F.sigmoid(self.w1)
        # gc = torch.add((1 - w1) * g1, w1 * output2)
        gc = torch.cat([g1, output2], dim=1)
        # gc = self.fc1(gc)
        # gc = self.relu(gc)
        # gc = self.dropout(gc)
        # gc = self.fc2(gc)
        # gc = self.relu(gc)
        # gc = self.dropout(gc)
        out = self.out(gc)
        output = F.softmax(out)
        return output


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
hidden = 8
dropout = 0.6
nb_heads = 8
alpha = 0.2
lr = 0.005
weight_decay = 5e-4
epochs = 10000
patience = 100
cuda = torch.cuda.is_available()
class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定
