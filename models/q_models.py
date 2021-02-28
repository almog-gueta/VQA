import torch
from torch import nn, Tensor
from abc import ABCMeta
from nets.fc import FCNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class lstm(nn.Module, metaclass=ABCMeta):
    """
    q model- embedding and lstm
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layer: int, num_hid, output_dim, activation, dropout: float, is_atten: bool, max_q_length: int):
        super(lstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if num_layer == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layer, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.is_atten = is_atten
        bidirectional = 2
        # if self.is_atten:
        #     self.fc = nn.Linear(hidden_dim * bidirectional, hidden_dim * bidirectional) # FCNet([hidden_dim * bidirectional, hidden_dim], activation, dropout=dropout)
        if not self.is_atten:
            self.fc = nn.Linear(hidden_dim * bidirectional * num_layer, output_dim) # FCNet([hidden_dim * bidirectional * num_layer, output_dim], activation, dropout=dropout)
        self.q_out_dim = hidden_dim * bidirectional if is_atten else output_dim

    def forward(self, q: tuple) -> Tensor:
        """
        Forward q through lstm
        :param q: q encoded
        :return: last lstm hidden on q after embedding and fc
        """
        # q=questions, q_lens: [batch_size, max_q_len], [batch_size, 1]
        questions = q[0]
        q_lens = q[1]
        emb = self.embedding(questions) # [batch_size, max_q_len, emb_dim]
        packed = pack_padded_sequence(emb, q_lens, batch_first=True)
        lstm_hid, (hn, cn) = self.lstm(packed) # hn = [num_layers * num_directions, batch_size, hidden_dim], lstm_hid.data = [batch_size, max_q_len, num_directions*hidden_dim]

        if self.is_atten:
            unpacked, _ = pad_packed_sequence(lstm_hid, batch_first=True)  # [batch_size, current_max_q_len, num_directions*hidden_dim]
            seq_len = questions.shape[1]  # max_q_len
            batch_size = questions.shape[0]  # batch_size
            if unpacked.shape[1] < seq_len:
                dummy_tensor = torch.zeros(batch_size, seq_len - unpacked.shape[1], unpacked.shape[2],
                                           requires_grad=lstm_hid.data.requires_grad, device=questions.device)
                lstm_hid = torch.cat([unpacked, dummy_tensor], dim=1)  # [batch_size, max_q_len, num_directions*hidden_dim]
            else:
                lstm_hid = unpacked # [batch_size, max_q_len, num_directions*hidden_dim]
            fc = lstm_hid #self.fc(lstm_hid) # [batch_size, max_q_len, num_directions*hidden_dim]
        else:
            hn = torch.transpose(hn, 0, 1)  # hn = [batch_size, num_layers * num_directions, hidden_size]
            fc = self.fc(hn.contiguous().view((hn.shape[0], -1))) # [batch_size, output_dim=1024]

        return fc


