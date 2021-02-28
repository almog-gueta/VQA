from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
import torch


class basic_lstm_cnn(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, q_model, v_model, activation, num_hid: int, dropout: float, is_concat: bool, output_dim: int):
        super(basic_lstm_cnn, self).__init__()
        self.q_model = q_model
        self.v_model = v_model
        self.is_concat = is_concat
        input_dim = self.q_model.q_out_dim + self.v_model.v_out_dim if self.is_concat else self.q_model.q_out_dim
        self.classifier = nn.Linear(input_dim, output_dim) #FCNet([input_dim, num_hid, output_dim], activation, dropout=dropout)
        self.activation = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: tuple) -> Tensor:
        """
        Forward x through basic_lstm_cnn
        :param x: tuple of (v, q) as tensors
        :return: tensor of probabilities for each possible answer
        """
        v, q = x[0], x[1]
        v_out = self.v_model(v) # [batch_size, 1024]
        q_out = self.q_model(q) # [batch_size, 1024]
        if self.is_concat:
            cat = torch.cat((v_out, q_out), dim=1) # [batch_size, v_out[1] + q_out[1]]
        else:
            cat = v_out * q_out # [batch, 1024]
        act = self.activation(cat)
        out = self.classifier(act) # [batch_size, output_dim=num_ans]
        return self.logsoftmax(out)


class atten_lstm_cnn(nn.Module, metaclass=ABCMeta):
    """
    lstm, cnn and then high order attention from Idan's paper
    """
    def __init__(self, q_model, v_model, activation, num_hid: int, dropout: float, is_concat: bool, output_dim: int, attention_model):
        super(atten_lstm_cnn, self).__init__()
        self.q_model = q_model
        self.v_model = v_model
        self.attention = attention_model
        self.is_concat = is_concat
        self.q_linear_for_muti = nn.Linear(self.q_model.q_out_dim, 1024)
        self.v_linear_for_muti = nn.Linear(self.v_model.v_out_dim, 1024)
        input_dim = self.q_model.q_out_dim + self.v_model.v_out_dim if self.is_concat else 1024
        self.classifier = FCNet([input_dim, output_dim], activation, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x) -> Tensor:
        """
        Forward x through atten_lstm_cnn
        :param x: tuple of (v, q) as tensors
        :return: tensor of probabilities for each possible answer
        """
        v, q = x[0], x[1]
        v_out = self.v_model(v) # [batch_size, conv_out_dim, resize_h / denominator * resize_w / denominator]
        q_out = self.q_model(q) # [batch_size, max_q_len, num_directions*hidden_dim]

        a_q, a_v = self.attention(q_out, v_out) # [batch_size, num_directions*hidden_dim], [batch_size, conv_out_dim]

        if self.is_concat:
            cat = torch.cat((a_q, a_v), dim=1) # [batch_size, num_directions*hidden_dim + conv_out_dim]
        else:
            q_out = self.q_linear_for_muti(a_q) # [batch_size, 1024]
            v_out = self.v_linear_for_muti(a_v) # [batch_size, 1024]
            cat = v_out * q_out # [batch, 1024]

        out = self.classifier(cat) # [batch_size, output_dim=num_ans]
        return self.logsoftmax(out)