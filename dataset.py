"""
Here, we create a custom dataset
"""
import operator
import re

import h5py
import numpy as np
import torch
import pickle
import json
from utils import types, main_utils, text_utils, vision_utils
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
from PIL import Image
from torchvision import transforms


class MyDataset(Dataset):
    """
    Custom VQA dataset.
    """
    def __init__(self, cfg, data_name, w2idx=None, idx2w=None, is_padding=True) -> None:
        super(MyDataset, self).__init__()
        # Set variables
        print(f'--------create {data_name} dataset---------')
        self.q_path = main_utils.get_path(cfg, dataset=data_name, q_or_a='q')
        self.a_path = main_utils.get_path(cfg, dataset=data_name, q_or_a='a')
        self.cfg_max_q_length = cfg['dataset']['max_q_length']
        self.data_name = data_name
        self.dataset_path = cfg['main']["paths"][f'{self.data_name}_dataset']

        # Get files
        with open(self.q_path, 'r') as fd:
            questions_json = json.load(fd)
        # with open(self.a_path, 'r') as fd:
        #     answers_json = json.load(fd)

        # get q from data
        print('we are at creating raw q')
        raw_questions = {q['question_id']: q['question'] for q in questions_json['questions']}

        # preprocess q
        print('we are at preprocess q')
        self.preprocessed_questions = self.preprocess_questions(raw_questions)

        # create vocab:
        if data_name == 'train':
            self.w2idx, self.idx2w = main_utils.init_questions_vocab(self.preprocessed_questions)
        else:
            if w2idx is None: # sanity check- check if we have idx, word mapping for val dataset
                print('missing vocab in MyDataset')
            self.w2idx, self.idx2w = w2idx, idx2w

        self.num_tokens = len(self.w2idx)
        self.max_q_length = self.get_max_question_length()

        # encode q and a
        print('we are at encoding q and a')
        self.questions = {q_id: self.encode_question(raw_q, is_padding=is_padding) for q_id, raw_q in self.preprocessed_questions.items()}
        self.answers, self.num_of_ans = text_utils.load_v2(self.data_name, cfg)

        # preprocess vision
        print('we are at preprocess vision')
        self.imgs_file_path = cfg['vision_utils'][f'{self.data_name}_file_path']
        self.img_id2idx = self.create_img_id2idx()

        # Load features
        print('we are at loading features')
        self.features = self._get_features()

        # Create list of entries
        print('we are at creating entries')
        self.entries = self._get_entries()

    def __getitem__(self, index: int) -> Tuple:
        v_idx = self.img_id2idx[self.entries[index][0]]
        imgs_file = h5py.File(self.imgs_file_path, mode='r')
        v = torch.from_numpy(imgs_file.get('imgs')[v_idx, :, :, :].astype('float32')) # [3, resize_h, resize_w]
        q = self.entries[index][1]  # [19], [1]
        a = self.entries[index][2]  # [num_ans]
        return (v, q, a)

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.entries)


    def preprocess_questions(self, questions_dict):
        _special_chars = re.compile('[^a-z0-9 ]*')

        for k, q in questions_dict.items():
            q = text_utils.process_digit_article(text_utils.process_punctuation(q))
            q = _special_chars.sub('', q)
            questions_dict[k] = q
        return questions_dict

    def get_max_question_length(self):
        data_max_length = 0
        for q in self.preprocessed_questions.values():
            length = len(q.split())
            if length > data_max_length:
                data_max_length = length
        max_q_length = min(self.cfg_max_q_length, data_max_length)
        return max_q_length

    def encode_question(self, question, is_padding=True):
        """
        Convert question from string (words) to torch vec of indices and return the vec
        If padding than len of vec is max_q_len, and indices initialized to '_pad' index
        :param question: question as a string (words)
        :param is_padding: flag if to pad vector or not
        :return: question as vector of indexes
        """
        q_words = question.split()
        if is_padding:
            vec = torch.zeros(self.max_q_length).long().fill_(self.w2idx['_pad'])
        else:
            vec = torch.zeros(len(q_words)).long()
        for i, token in enumerate(q_words):
            if i >= self.max_q_length:
                break
            index = self.w2idx.get(token, self.w2idx['_unk'])
            vec[i] = index
        return vec, min(len(q_words), self.max_q_length)


    def create_img_id2idx(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.imgs_file_path, mode='r') as imgs_file:
            img_ids = imgs_file['img_ids'][()]
        img_id2idx = {id: i for i, id in enumerate(img_ids)}
        return img_id2idx

    def _save(self):
        torch.save(self, self.dataset_path)


    def _get_features(self) -> Any:
        """
        Load all features into a structure (not necessarily dictionary). Think if you need/can load all the features
        into the memory.
        :return:
        :rtype:
        """
        # with open(self.path, "rb") as features_file:
        #     features = pickle.load(features_file)
        #
        # return features
        pass

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        entries = []

        for item in self.answers:
            a = item['label_scores']
            if self.data_name == 'train' and len(a) == 0:
                continue
            a = torch.tensor([a.get(x, 0.0) for x in range(self.num_of_ans)], requires_grad=False)  # [num_of_ans]
            q_id = item['question_id']
            q = self.questions[q_id] # q, len_q
            v_id = item['image_id']
            entries.append((v_id, q, a))

        return entries
