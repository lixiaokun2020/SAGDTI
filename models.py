from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import MessagePassing
import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)


class SAGDTI(nn.Sequential):
    '''
        Molecular information with 2D interaction pairing map
        Biological interactive information  with graph attention network
    '''
    
    def __init__(self, nd_layer, nt_layer, GATlayer, filters, **config):
        super(SAGDTI, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        
        #densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = torch.cuda.device_count()


        #molecular transformer encoder
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.nd_layer = nd_layer
        self.nt_layer = nt_layer

        self.flatten_dim = config['flat_dim']

        #Biological attributes with GAT
        self.graphdim = config['graphdim']
        self.GAThid1 = config['GAThid1']
        self.GAThid2 = config['GAThid2']
        self.weight_decay = config['weight_decay']
        self.GATlayers = GATlayer
        
        # specialized embedding with positional one
        self.demb = PositionAwareAttention()
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        
        self.d_encoder = Mol_trans_encoder_MultipleLayers(self.nd_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        self.p_encoder = Mol_trans_encoder_MultipleLayers(self.nt_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)


        self.mol_cnn = nn.Conv2d(1, filters, 3, padding = 0)



        self.Bio_att = GAT_MultipleLayers()

        
        self.decoder = nn.Sequential(
            nn.Conv2d(1,filters,3,padding=0),
            nn.MaxPool2d(2,1),

            nn.Conv2d(1, filters, 3, padding=0),
            nn.MaxPool2d(2,1),

            nn.Conv2d(1, filters, 3, padding=0),
            nn.MaxPool2d(2,1),

            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        )



        
    def forward(self, d,p,feature, graph, graph2adj):

        d_emb = self.demb(d) # batch_size x seq_length x embed_size
        p_emb = self.pemb(p)



        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        
        d_encoded_layers = self.d_encoder(d_emb.float(), self.attention_probs_dropout_prob)
        #print(d_encoded_layers.shape)
        p_encoded_layers = self.p_encoder(p_emb.float(), self.attention_probs_dropout_prob)

        Bio_att = self.Bio_att(feature, graph, graph2adj)
        #print(p_encoded_layers.shape)

        # repeat to have the same tensor size for aggregation   
        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1) # repeat along protein size
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1) # repeat along drug size
        
        i = d_aug * p_aug # interaction
        i_v = i.view(int(self.batch_size), -1, self.max_d, self.max_p)
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim = 1)

        #print(i_v.shape)
        i_v = torch.unsqueeze(i_v, 1)
        #print(i_v.shape)
        
        i_v = F.dropout(i_v, p = self.dropout_rate)        
        
        #f = self.mol_cnn2(self.mol_cnn1(i_v))
        f = self.mol_cnn(i_v)

        
        #print(f.shape)

        #f = self.dense_net(f)
        #print(f.shape)
        
        mol_att = f.view(int(self.batch_size), -1)

        X = torch.stack([mol_att, Bio_att], dim=1)

        score = self.decoder(X)
        return score    
   
# help classes    


class GAL(MessagePassing):
    def __init__(self, in_features, out_featrues):
        super(GAL, self).__init__()
        self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_featrues, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        # 定义leakyrelu激活函数
        self.leakyrelu = torch.nn.LeakyReLU()
        self.linear = torch.nn.Linear(in_features, out_featrues)

    def forward(self, x, edge_index):
        x = self.linear(x)
        N = x.size()[0]
        row, col = edge_index
        a_input = torch.cat([x[row], x[col]], dim=1)

        temp = torch.mm(a_input, self.a).squeeze()
        e = self.leakyrelu(temp)
        e_all = torch.zeros(N)
        for i in range(len(row)):
            e_all[row[i]] += math.exp(e[i])

        # f = open("atten.txt", "w")

        for i in range(len(e)):
            e[i] = math.exp(e[i]) / e_all[row[i]]
        #     f.write("{:.4f}\t {} \t{}\n".format(e[i], row[i], col[i]))
        #
        # f.close()
        return self.propagate(edge_index, x=x, norm=e)

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j

class GATLay(torch.nn.Module):
    def __init__(self, in_features, hid_features, out_features, n_heads):
        super(GATLay, self).__init__()
        self.attentions = [GAL(in_features, hid_features) for _ in
                           range(n_heads)]
        self.out_att = GAL(hid_features * n_heads, out_features)

    def forward(self, x, edge_index, dropout):
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return F.softmax(x, dim=1)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size, opt):
        super(PositionAwareAttention, self).__init__()

        self.opt = opt
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=True)

        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=True)
        else:
            self.wlinear = None

        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):

        # TODO: experiment with he and xavier
        # done, not really helping in any way here
        self.ulinear.weight.data.normal_(std=0.001).to("cuda")
        self.vlinear.weight.data.normal_(std=0.001).to("cuda")
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001).to("cuda")

        self.tlinear.weight.data.zero_().to("cuda")  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f, lstm_units=None, lstm_layer=False):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """

        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)

        # TODO: vlinear vs ulinear, u works better, but does it make sense to share those weights?
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
            batch_size, seq_len, self.attn_size
        )

        """
        # q_proj done step by step here to catch errors better
        # info on view and unsqueeze - 
        # https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155
        # q.size = 50x200
        q_proj = self.vlinear(q.view(-1, self.query_size))  # 50x200
        q_proj = q_proj.contiguous()  # 50x200
        q_proj = q_proj.view(batch_size, self.attn_size)  # <-- this is were size error happens  # 50x200
        q_proj = q_proj.unsqueeze(1)  # 50x200, adds new dimension
        q_proj = q_proj.expand(batch_size, seq_len, self.attn_size)  # 50x91x200 batch x seq_size x hidden size
        """

        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size
            )
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]

        # view in PyTorch is like reshape in numpy, view(n_rows, n_columns)
        # view(-1, n_columns) - here we define the number of columns, but n_rows will be chosen by PyTorch
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(batch_size, seq_len)

        # mask padding
        # print(x_mask, x_mask.size())
        # print(x_mask.data)

        # fill elements of self tensor with value where mask is one
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=-1)

        # weighted average input vectors

        # to calculate final sentence representation z,
        # we first test two variants:

        # 1. use self-attention to calculate a_i and use lstm to get h_i
        if lstm_layer:
            outputs = weights.unsqueeze(1).bmm(lstm_units).squeeze(1)
        # 2. use self-attention for a_i and also for h_i
        else:
            outputs = weights.unsqueeze(1).bmm(x).squeeze(1)

        return outputs




class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Mol_trans_encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Mol_trans_encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output    

    
class Mol_trans_encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Mol_trans_encoder_MultipleLayers, self).__init__()
        layer = Mol_trans_encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GAT, self).__init__()
        self.gatlay = GATLay(nfeat, nhid, nhid2, 4)
        self.dropout = dropout

    def forward(self, x, adj):
        edge_index = adj
        x = self.gatlay(x, edge_index, self.dropout)
        return x

class GAT_MultipleLayers(nn.Module):
    def __init__(self, n_layer, nfeat, nhid1, nhid2, dropout):
        super(GAT_MultipleLayers, self).__init__()
        layer = GAT(nfeat, nhid1, nhid2, dropout)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, x, adj):
        for layer_module in self.layer:
            hidden_states = layer_module(x, adj)

        return hidden_states
