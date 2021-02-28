"""
Main utils file, all utils functions that are not related to train.
"""
import json
import os
import hydra
import torch
import schema
import operator
import functools
from torch import nn
import torch.utils.data as data
from typing import Dict
from utils.types import PathT
from collections import MutableMapping
from utils.config_schema import CFG_SCHEMA
from omegaconf import DictConfig, OmegaConf

from models import q_models, v_models, vqa_model, attentions


def get_path(cfg, dataset, q_or_a):
    """
    Return the path of the desired dataset of VQA
    :param dataset: train or val
    :param q_or_a: questions or annotations
    :return: path to the desired file
    """

    if dataset == 'train':
        data = 'train2014'
    else:
        data = 'val2014'

    if q_or_a == 'q':
        fmt = 'v2_{0}_{1}_{2}_questions.json'
    else:
        fmt = 'v2_{1}_{2}_annotations.json'

    relative_path = fmt.format(cfg['main_utils']['task'], cfg['main_utils']['dataset'], data)
    return os.path.join('/', cfg['main_utils']['qa_path'], relative_path)

def init_questions_vocab(questions_dict):
    vocab = set()
    for q in questions_dict.values():
        for w in q.split():
            vocab.add(w)

    special_tokens = ['_pad', '_unk', '_mask']
    w2idx = {}
    idx2w = {}
    for i, t in enumerate(special_tokens):
        w2idx[t] = i
        idx2w[i] = t

    for i, w in enumerate(vocab):
        idx = i + len(special_tokens)
        w2idx[w] = idx
        idx2w[idx] = w
    return w2idx, idx2w

class init_models():
    def __init__(self, q_model_name, v_model_name, vqa_model_name, cfg, max_q_length, num_of_ans, model_name=None) -> None:
        self.q_model = self.init_q_model(q_model_name, cfg, max_q_length)
        self.v_model = self.init_v_model(v_model_name, cfg, model_name)
        self.model = self.init_vqa_model(q_model_name, v_model_name, vqa_model_name, cfg, num_of_ans, max_q_length)

    def init_q_model(self, q_model_name, cfg, max_q_length):
        q_model_params = dict(cfg['q_model'][q_model_name], **{'max_q_length': max_q_length})
        if q_model_name == 'lstm' or q_model_name == 'attention_lstm':
            q_model = q_models.lstm(**q_model_params)
        return q_model

    def init_v_model(self, v_model_name, cfg, model_name=None):
        resizes_dict = {'resize_h': cfg['dataset']['resize_h'], 'resize_w': cfg['dataset']['resize_w']}
        v_model_params = dict(cfg['v_model'][v_model_name], **resizes_dict)
        if v_model_name == 'cnn' or v_model_name == 'attention_cnn':
            v_model = v_models.CNN(**v_model_params)

        # if using autoEncoder pretrained CNN than load state dict as written below:
        # for pretrained autoEncoder with 8 CNN layers, use this line:
        if model_name == 'pretrain_8_layers':
            # model_dict = torch.load("/home/student/hw2/autoencoder_8_layers/trained_cnn_150.pth", map_location=lambda storage, loc: storage)
            model_dict = torch.load("./autoencoder_saved_models/autoencoder_8_layers.pth",
                                    map_location=lambda storage, loc: storage)

        # for pretrained autoEncoder with 4 CNN layers, use this line:
        if model_name == 'pretrain_4_layers':
            # model_dict = torch.load("/home/student/hw2/autoencoder_4_layers/trained_cnn_80.pth", map_location=lambda storage, loc: storage)
            model_dict = torch.load("./autoencoder_saved_models/autoencoder_4_layers.pth",
                                    map_location=lambda storage, loc: storage)

        # for both models- use the below lines:
        if model_name == 'pretrain_8_layers' or model_name == 'pretrain_4_layers':
            if v_model_params["is_atten"] == True and "fc.weight" in model_dict.keys():
                del model_dict["fc.weight"]
                del model_dict["fc.bias"]
            v_model.load_state_dict(model_dict)

        return v_model

    def init_atten_model(self, q_model_name, v_model_name, cfg, max_q_length):
        emb_Q_dim = cfg['q_model'][q_model_name]['hidden_dim'] * 2 # bidirectional
        emb_I_dim = cfg['v_model'][v_model_name]['dims'][-1]
        projected_dim = cfg['atten_model']['projected_dim']
        return attentions.high_order_attention(emb_Q_dim, emb_I_dim, projected_dim, max_q_length, num_regions = self.v_model.fc_out)

    def init_vqa_model(self, q_model_name, v_model_name, vqa_model_name, cfg, num_of_ans, max_q_length):
        vqa_params = dict(cfg['vqa_model'][vqa_model_name], **{'output_dim': num_of_ans})
        model_params = dict({'q_model': self.q_model, "v_model": self.v_model}, **vqa_params)
        if vqa_model_name == 'basic_lstm_cnn':
            model = vqa_model.basic_lstm_cnn(**model_params)
        elif vqa_model_name == 'atten_lstm_cnn':
            atten_model = self.init_atten_model(q_model_name, v_model_name, cfg, max_q_length)
            model_params = dict(model_params, **{"attention_model": atten_model})
            model = vqa_model.atten_lstm_cnn(**model_params)
        return model

def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[1][-1], reverse=True)
    return data.dataloader.default_collate(batch)


def get_model_string(model: nn.Module) -> str:
    """
    This function returns a string representing a model (all layers and parameters).
    :param model: instance of a model
    :return: model \n parameters
    """
    model_string: str = str(model)

    n_params = 0
    for w in model.parameters():
        n_params += functools.reduce(operator.mul, w.size(), 1)

    model_string += '\n'
    model_string += f'Params: {n_params}'

    return model_string


def set_seed(seed: int) -> None:
    """
    Sets a seed
    :param seed: seed to set
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(path: PathT) -> None:
    """
    Given a path, creating a directory in it
    :param path: string of the path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def warning_print(text: str) -> None:
    """
    This function prints text in yellow to indicate warning
    :param text: text to be printed
    """
    print(f'\033[93m{text}\033[0m')


def validate_input(cfg: DictConfig) -> None:
    """
    Validate the configuration file against schema.
    :param cfg: configuration file to validate
    """
    cfg_types = schema.Schema(CFG_SCHEMA)
    cfg_types.validate(OmegaConf.to_container(cfg))


def _flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a dictionary.
    For example:
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]} ->
    {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}
    :param d: dictionary to flat
    :param parent_key: key to start from
    :param sep: separator symbol
    :return: flatten dictionary
    """
    items = []

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def get_flatten_dict(cfg: DictConfig) -> Dict:
    """
    Returns flatten dictionary, given a config dictionary
    :param cfg: config file
    :return: flatten dictionary
    """
    return _flatten_dict(cfg)


def init(cfg: DictConfig) -> None:
    """
    :cfg: hydra configuration file
    """
    # TODO: Trains
    os.chdir(hydra.utils.get_original_cwd())
    validate_input(cfg)