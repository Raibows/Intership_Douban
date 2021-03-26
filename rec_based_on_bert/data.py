from torch.utils.data import Dataset
from tqdm import tqdm
from .transformers import BertTokenizer
from .config import douban_dataset_config, model_config, DEBUG_MODE
import torch
import torch.nn.functional as F
import random

random.seed(6)

def read_douban_data(path, threshold=douban_dataset_config.label_threshold,
                     label_keeps=None):
    datas = []
    labels = []
    label_uniq = {}
    label_cnt = {}
    keys = set()
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                # print(line)
                continue
            line = line.split('\t')
            if line[0] in keys or line[0] == 'NULL':
                continue
            keys.add(line[0])
            datas.append(line[-1].strip('\n'))
            labels.append((line[3].split('/'), line))
            for l in labels[-1][0]:
                if l not in label_uniq:
                    label_cnt[l] = 0
                    label_uniq[l] = len(label_uniq)
                label_cnt[l] += 1
    # print(label_cnt)
    if not label_keeps:
        for key in label_uniq.copy().keys():
            if label_cnt[key] < threshold * sum(label_cnt.values()) / len(label_cnt):
                del label_cnt[key]
                del label_uniq[key]
        t = 0
        for key in label_uniq.keys():
            label_uniq[key] = t
            t += 1
    else:
        label_uniq = label_keeps

    new_datas = []
    new_labels = []
    name_dict = {}
    for i in range(len(datas)):
        temp = []
        for l in labels[i][0]:
            if l in label_uniq:
                temp.append(l)
        if len(temp) > 0:
            new_datas.append(datas[i])
            new_labels.append(labels[i][0])
            name_dict[labels[i][1][1]] = (len(new_datas) - 1, labels[i][1])

    mean_len = 0
    for x in new_datas:
        mean_len += len(x)

    print(f'num all is {len(new_datas)}, mean len is {mean_len / len(new_datas)}')


    print(label_cnt)
    print(label_uniq)
    limit = -1 if not DEBUG_MODE else 20
    return new_datas[:limit], new_labels[:limit], label_uniq, int(mean_len/len(new_datas)), name_dict

def split_to(path, rate, path1, path2):
    with open(path, 'r') as file:
        origins = file.readlines()
    temp = [i for i in range(len(origins))]
    random.shuffle(temp)
    rate = int(rate * len(origins))
    with open(path1, 'w') as file:
        for i in temp[:rate]:
            file.write(origins[i])

    with open(path2, 'w') as file:
        for i in temp[rate:]:
            file.write(origins[i])

def concat_csv(target):
    paths = ['dzz', 'wzy', 'hxj', 'nys', 'xmy']
    temp = []
    for i, path in enumerate(paths):
        path = f'./dataset/douban_{path}.csv'
        with open(path, 'r') as file:
            for j, line in enumerate(file):
                if i == 0 and j == 0:
                    temp.append(line)
                elif j > 0:
                    temp.append(line)

    with open(target, 'w') as file:
        for line in temp:
            file.write(line)

class douban_dataset(Dataset):
    def __init__(self, data_path, label_keeps=None):
        self.datas, self.labels, self.label_uniq, self.mean_len, self.name_dict  \
            = read_douban_data(data_path, label_keeps=label_keeps)
        self.tokenizer = BertTokenizer.from_pretrained(model_config.bert_type)
        self.datas_tensor, self.labels_tensor = self.convert_to_ids()
        self.label_num = len(self.label_uniq)
        self.inverse_list = self.build_inverse()

    def convert_to_ids(self):
        labels_tensor = []
        for yy in self.labels:
            temp = torch.zeros([len(self.label_uniq)], dtype=torch.float)
            for y in yy:
                if y not in self.label_uniq: continue
                temp[self.label_uniq[y]] = 1
            labels_tensor.append(temp)

        datas_tensor = {
            'inputs': [],
            'types': [],
            'masks': [],
        }
        temp = self.tokenizer(self.datas, max_length=int(self.mean_len*1.3),
                              truncation=True, padding='max_length')
        for i in range(len(self.datas)):
            datas_tensor['inputs'].append(torch.tensor(temp['input_ids'][i], dtype=torch.long))
            datas_tensor['types'].append(torch.tensor(temp['token_type_ids'][i], dtype=torch.long))
            datas_tensor['masks'].append(torch.tensor(temp['attention_mask'][i], dtype=torch.long))

        return datas_tensor, labels_tensor

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return (self.datas_tensor['inputs'][item], self.datas_tensor['types'][item],
                self.datas_tensor['masks'][item]), self.labels_tensor[item]

    def __setitem__(self, key, value):
        self.datas_tensor['inputs'][key], self.datas_tensor['types'][key], \
        self.datas_tensor['masks'][key] = value[0]
        self.labels_tensor[key] = value[1]

    def build_inverse(self):
        temp = [0 for _ in self.label_uniq.keys()]
        for k, v in self.label_uniq.items():
            temp[v] = k
        return temp



    def index_by_movie_name(self, name):
        if name in self.name_dict:
            item = self.name_dict[name][0]
            return (self.datas_tensor['inputs'][item], self.datas_tensor['types'][item],
                self.datas_tensor['masks'][item]), self.name_dict[name][1]
        else:
            return None, None

    def inverse_index_to_genre(self, index:list):
        temp = {}
        for i in range(len(index)):
            temp[self.inverse_list[i]] = index[i]
        return temp




if __name__ == '__main__':
    split_to('./dataset/all.csv', 0.8, './dataset/train.csv', './dataset/test.csv')
    # dataset = douban_dataset('./dataset/all.csv')
    # concat_csv('./dataset/all.csv')
    pass