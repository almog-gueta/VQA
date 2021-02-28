from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
import torch.nn.functional as F
import torch



class entity_info(nn.Module, metaclass=ABCMeta):
    """
    calc attention between object (word or region) to itself
    """

    def __init__(self, input_dim, output_dim, activation, dropout: float):
        super(entity_info, self).__init__()
        self.fc = FCNet([input_dim, input_dim, output_dim], activation, dropout=dropout)

    def forward(self, x: torch) -> Tensor:
        return self.fc(x)


class dot_product_message(nn.Module, metaclass=ABCMeta):
    """
    calc attention between Q and I
    """

    def __init__(self, emb_Q_dim, emb_I_dim, projected_dim, num_words, num_regions):
        super(dot_product_message, self).__init__()
        self.emb_q = nn.Linear(emb_Q_dim, projected_dim) # emb_Q_dim = 1024
        self.emb_i = nn.Linear(emb_I_dim, projected_dim) # emb_I_dim = cnn_out2
        self.linear_q = nn.Linear(num_regions , 1) # [num_words, num_regions] -> [num_words, 1]
        self.linear_i = nn.Linear(num_words, 1)# [num_regions, num_words] -> [num_regions, 1]


    def forward(self, Q, I) -> tuple:
        # Q = [batch_size, num_words, repres_dim]
        # I = [batch_size, num_regions, cnn_out2]

        # normalize Q= [num_words, repres_dim] 19, 1024
        emb_q = self.emb_q(Q)
        norm_Q = emb_q / emb_q.norm(dim=0)

        # normalize I= [num_regions, repres_dim] 56*56=3136, cnn_out2
        emb_i = self.emb_i(I)
        norm_I = emb_i / emb_i.norm(dim=0)

        # dot product Q*I
        matrix = torch.matmul(norm_Q, torch.transpose(norm_I,1,2)) # [num_words, num_regions]
        # weighted marginalize for Q and I
        q_out = self.linear_q(matrix)
        i_out = self.linear_i(torch.transpose(matrix,1, 2))

        return q_out, i_out



class high_order_attention(nn.Module, metaclass=ABCMeta):

    def __init__(self, emb_Q_dim, emb_I_dim, projected_dim, num_words, num_regions, weights_dim = 1):
        super(high_order_attention, self).__init__()

        self.q_w_I = nn.Linear(weights_dim, weights_dim, bias=False)
        self.q_w_II = nn.Linear(weights_dim, weights_dim, bias=False)
        self.q_w_IQ = nn.Linear(weights_dim, weights_dim, bias=False)
        self.i_w_I = nn.Linear(weights_dim, weights_dim, bias=False)
        self.i_w_II = nn.Linear(weights_dim, weights_dim, bias=False)
        self.i_w_IQ = nn.Linear(weights_dim, weights_dim, bias=False)
        self.q_entity_info = entity_info(emb_Q_dim, output_dim=1, activation='ReLU', dropout=0.0)
        self.i_entity_info = entity_info(emb_I_dim, output_dim=1, activation='ReLU', dropout=0.0)
        self.Q2I_message = dot_product_message(emb_Q_dim, emb_I_dim, projected_dim, num_words, num_regions)
        self.q_self_message = dot_product_message(emb_Q_dim, emb_Q_dim, emb_Q_dim, num_words, num_words)
        self.i_self_message = dot_product_message(emb_I_dim, emb_I_dim, emb_I_dim, num_regions, num_regions)

    def forward(self,Q, I) -> tuple:
        # Q [batch_size, repres_dim 1024] OR [batch_size, max_q_len, num_directions*hidden_dim]
        # I [batch_size, repres_dim 1024] OR [batch_size, conv_out_l2, resize_h / 4 * resize_w / 4]

        I = torch.transpose(I,1,2) # [batch_size, resize_h/4 * resize_w/4, conv_out_l2]

        entity_q = self.q_entity_info(Q) # [batch_size, max_q_len, 1]
        entity_i = self.i_entity_info(I) # [batch_size, resize_h/4 * resize_w/4, 1]

        q2i_message, i2q_message = self.Q2I_message(Q, I) # [batch_size, max_q_len, 1], [batch_size, resize_h/4 * resize_w/4, 1]

        self_q, _ = self.q_self_message(Q, Q) # [batch_size, max_q_len, 1]
        self_i, _ = self.i_self_message(I, I) # [batch_size, resize_h/4 * resize_w/4, 1]

        belief_q = F.softmax(self.q_w_I(entity_q) + self.q_w_II(self_q) + self.q_w_IQ(q2i_message), dim=1) # [batch_size, max_q_len, 1]
        belief_i = F.softmax(self.i_w_I(entity_i) + self.i_w_II(self_i) + self.i_w_IQ(i2q_message), dim=1) # [batch_size, resize_h/4 * resize_w/4, 1]

        # [batch_size, 1, max_q_len]*[batch_size, max_q_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        a_q = torch.bmm(torch.transpose(belief_q, 1, 2), Q).squeeze(1)
        # [batch_size, 1, resize_h/4 * resize_w/4]*[batch_size, resize_h/4 * resize_w/4, conv_out_l2] = [batch_size, 1, conv_out_l2] -> [batch_size, conv_out_l2]
        a_i = torch.bmm(torch.transpose(belief_i, 1, 2), I).squeeze(1)

        return a_q, a_i





