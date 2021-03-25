'''
基于课上讲的贝叶斯算法原理，实现3分类或者支持任意分类的贝叶斯算法，3分类必做，通用任意分类（选做）。
'''

import numpy as np
import random


train_path = '/home/jsjlab/projects/AttackViaGan/dataset/AGNEWS/train.std'
test_path = '/home/jsjlab/projects/AttackViaGan/dataset/AGNEWS/test.std'
label_dict = {}
label_inverse = []

def read_standard_data(path):
    data = []
    labels = []
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip('\n')
            data.append(line[:-1])
            label = int(line[-1])
            if label not in label_dict:
                label_dict[label] = len(label_dict)
                label_inverse.append(label)
            labels.append(label_dict[label])
    print(f'loading data {len(data)} from {path}')
    return {'x':data, 'y': labels}

def read_datas(path):
    labels = []
    datas = []
    with open(path, 'r', encoding='latin-1') as file:
        for line in file:
            line = line.strip('\n')
            temp = line.split(',')
            if temp[0] not in label_dict:
                label_dict[temp[0]] = len(label_dict)
                label_inverse.append(temp[0])
            labels.append(label_dict[temp[0]])
            datas.append(line[len(temp[0])+1:-3].split(' '))

    return datas, labels

def split_dataset(datas, labels, split_rate):
    assert len(datas) == len(labels)
    split_rate = int(len(datas) * split_rate)
    chosen = random.sample([i for i in range(len(datas))], k=split_rate)
    chosen = set(chosen)
    datas_A = {'x': [], 'y': []}
    datas_B = {'x': [], 'y': []}
    for i in range(len(datas)):
        if i in chosen:
            datas_A['x'].append(datas[i])
            datas_A['y'].append(labels[i])
        else:
            datas_B['x'].append(datas[i])
            datas_B['y'].append(labels[i])

    return datas_A, datas_B

def build_word_dict(datas):
    word_dict = {}
    for sen in datas:
        for word in sen:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict

def tokenize(word_dict, sen):
    ids = []
    unk_token = len(word_dict)
    for word in sen:
        if word not in word_dict:
            ids.append(unk_token)
        else:
            ids.append(word_dict[word])
    return ids

class naive_bayes_model():
    def __init__(self, train_datas, train_labels, word_dict, label_dict):
        self.label_num = len(label_dict)
        self.word_num = len(word_dict)
        self.train_datas = train_datas
        self.train_labels = train_labels
        self.unk_token_id = self.word_num
        self.label_count = [0 for _ in range(self.label_num)]
        self.word_count = [[1 for _ in range(self.word_num+1)] for _ in range(self.label_num)]

        self.init_counts()
        
    
    def init_counts(self):
        for i in range(len(self.train_datas)):
            self.label_count[self.train_labels[i]] += 1
            for w in self.train_datas[i]:
                self.word_count[self.train_labels[i]][w] += 1
        self.label_count = np.array(self.label_count, dtype=np.float32) / len(self.train_datas)
        self.word_count = np.array(self.word_count, dtype=np.float32).T
        self.word_all = np.array(
            [np.sum(s) for s in self.word_count.T], dtype=np.float32
        )
    
    def __predict(self, sen_idx:[int]):
        # ln( p(y_i) * prod_j ( p(w_j | y_i) ) )
        probs = np.log(self.label_count)
        for w in sen_idx:
            probs += np.log(self.word_count[w] / self.word_all)

        return np.argmax(probs)

    def predict_on_dataset(self, datas, datas_id, labels, verbose=False):
        assert len(datas_id) == len(labels)
        all_num = len(labels)
        correct = all_num
        for i, sen in enumerate(datas_id):
            p = self.__predict(sen)
            true_label = labels[i]
            if p != true_label:
                correct -= 1
                if verbose:
                    print(f'origin label: {label_inverse[true_label]} predicts wrong: {label_inverse[p]}')
                    print(' '.join(datas[i]))
                    print('---' * 10)

        return correct / all_num


if __name__ == '__main__':
    # datas, labels = read_datas(data_path)
    # print(label_dict)
    # test_dataset, train_dataset = split_dataset(datas, labels, 0.2)
    train_dataset = read_standard_data(train_path)
    test_dataset = read_standard_data(test_path)

    train_size = len(train_dataset['x'])
    test_size = len(test_dataset['x'])

    print(f"the train datasize is {train_size}, test dataset size is {test_size}, "
          f"test dataset split ratio is {test_size/(train_size+test_size):.5f}")


    word_dict = build_word_dict(train_dataset['x'])
    train_dataset['x_ids'] = [tokenize(word_dict, sen) for sen in train_dataset['x']]
    test_dataset['x_ids'] = [tokenize(word_dict, sen) for sen in test_dataset['x']]

    model = naive_bayes_model(train_dataset['x_ids'], train_dataset['y'], word_dict, label_dict)

    train_acc = model.predict_on_dataset(train_dataset['x'], train_dataset['x_ids'], train_dataset['y'])
    test_acc = model.predict_on_dataset(test_dataset['x'], test_dataset['x_ids'], test_dataset['y'])

    print(f'on train dataset the accuracy is {train_acc:.5f}')
    print(f'on test dataset the accuracy is {test_acc:.5f}')





