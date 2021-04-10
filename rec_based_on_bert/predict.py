import argparse
import copy
import re
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data import douban_dataset
from config import model_config, douban_dataset_config, DEBUG_MODE
from model import bert_douban
import numpy as np
from datetime import datetime
import random
from metric import metric_accuracy

def init():
    def setup_seed(seed):
        print(f'setup seed {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(6)

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')


    data_base = douban_dataset(douban_dataset_config.train_path)
    model = bert_douban(model_config.bert_type, model_config.fc_hidden_size, model_config.fc_num_layer,
                        data_base.label_num, model_config.fc_dropout_rate, model_config.bert_fine_tune)
    model.load_state_dict(torch.load(model_config.model_load_path, map_location=device))
    print(f'loading model from {model_config.model_load_path}')
    model = model.to(device)
    return model, data_base, device

@torch.no_grad()
def predict_movie_genres(model, data_base, device, name):
    model.eval()
    X, info = data_base.index_by_movie_name(name)
    if X and info:
        temp = [t.unsqueeze(0).to(device) for t in X]
        logits = torch.sigmoid(model(temp))[0].tolist()
        print(logits)
        preds = data_base.inverse_index_to_genre(logits)
        return preds, info
    return None, None


if __name__ == '__main__':
    # test_movie_name = '大护法'
    # logits, info = predict_movie_genres(test_movie_name)
    # print(logits)
    # print(info)
    pass
