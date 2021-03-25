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


def parse_bool(v):
    return 'y' in v.lower()

def setup_seed(seed):
    print(f'setup seed {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--load_model', choices=[True, False], default='no', type=parse_bool)
parser.add_argument('--cuda', type=str, default='3')
parser.add_argument('--only_evaluate', type=parse_bool, default='no')
parser.add_argument('--valid_rate', type=float, default=0.2)
args = parser.parse_args()



setup_seed(6)
batch = args.batch if not DEBUG_MODE else 10
lr = args.lr
is_load_model = args.load_model
if args.cuda == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+args.cuda)




train_dataset = douban_dataset(douban_dataset_config.train_path)
test_dataset = douban_dataset(douban_dataset_config.test_path)
label_num = train_dataset.label_num


model = bert_douban(model_config.bert_type, model_config.fc_hidden_size, model_config.fc_num_layer,
                    train_dataset.label_num, model_config.fc_dropout_rate, model_config.bert_fine_tune)



test_dataset = DataLoader(test_dataset, batch_size=batch)

if is_load_model:
    model.load_state_dict(torch.load(model_config.model_load_path, map_location=device))
    print(f'loading model from {model_config.model_load_path}')
model = model.to(device)



optimizer = optim.AdamW(
    [
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': lr}
    ], lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5
)

    # optimizer = optim.Adam(model.net.parameters(), lr=lr, )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=3,
                                                     verbose=True, min_lr=3e-8)
warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < 4 else 1.0)

criterion = nn.BCEWithLogitsLoss()


def k_fold_split(dataset:douban_dataset, split_ratio:float):
    all = len(dataset)
    split = int(all * split_ratio)
    left = 0
    right = left + split
    kfolds = []
    random.shuffle(dataset)
    while True:
        if right > all: right = all
        temp = list(range(left, right))
        test = torch.utils.data.dataset.Subset(dataset, temp)
        temp = list(range(0, left))+list(range(right, all))
        train = torch.utils.data.dataset.Subset(dataset, temp)
        kfolds.append((
            DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(test, batch_size=batch)
        ))
        if right == all: break
        left += split
        right += split
    return kfolds


def train(train_dataset):
    loss_mean = 0.0
    model.train()
    with tqdm(total=len(train_dataset), desc='train') as pbar:
        for (x1, x2, x3), y in train_dataset:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            logits = model((x1, x2, x3))
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f'loss {loss.item():.4f}')
            pbar.update(1)
            loss_mean += loss.item()
    return loss_mean / len(train_dataset)

@torch.no_grad()
def evaluate(validation_dataset):
    model.eval()
    loss_mean = 0.0
    correct = 0
    num = 0
    with tqdm(total=len(validation_dataset), desc='test') as pbar:
        for (x1, x2, x3), y in validation_dataset:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            logits = model((x1, x2, x3))
            loss = criterion(logits, y).item()
            logits = torch.sigmoid(logits)
            logits[logits >= 0.5] = 1
            num += y.shape[0]
            correct += logits.eq(y).sum().item()
            loss_mean += loss
            pbar.set_postfix_str(f'loss {loss:.4f} acc{correct/num/label_num:.4f}')
            pbar.update(1)

    return loss_mean / len(validation_dataset), correct / num / label_num


def main():
    base_path = "./static/{}_{:.5f}_{:.5f}_bert_douban.pt"
    best_path = None
    best_state = None
    best_acc = 0.0
    best_loss = 1e9
    if is_load_model:
        temp = re.findall("\d+\.\d+", model_config.model_load_path)
        best_loss = float(temp[0])
        best_acc = float(temp[1])
    epoch = args.epoch

    dataset_kfolds = k_fold_split(train_dataset, split_ratio=args.valid_rate)
    fold_num = len(dataset_kfolds)
    folds_range = [i for i in range(fold_num)]

    for ep in range(epoch):
        train_loss = 0.0
        eval_loss = 0.0
        acc = 0.0
        random.shuffle(folds_range)
        for kf in folds_range:
            print(f'epoch {ep} fold {kf} training')
            train_loss += train(dataset_kfolds[kf][0])
            print(f'epoch {ep} fold {kf} evaluating')
            t1, t2 = evaluate(dataset_kfolds[kf][1])
            eval_loss += t1
            acc += t2

        train_loss /= fold_num
        eval_loss /= fold_num
        acc /= fold_num

        if ep < 4:
            warmup_scheduler.step(ep)
        else:
            scheduler.step(eval_loss)

        if acc > best_acc:
            best_loss = eval_loss
            best_acc = acc
            best_path = base_path.format(datetime.now().strftime("%m_%d_%H_%M_%S"), best_loss, best_acc)
            best_state = copy.deepcopy(model.state_dict())



        print(f'epoch {ep} done! train_loss {train_loss:.5f}  '
                f'eval_loss {eval_loss:.5f} best acc {best_acc:.5f} best loss {best_loss:.5f}')
    if best_state:
        torch.save(best_state, best_path)
        print(f'saving model to {best_path}')

    print(f'the best model have been saved in {best_path}')





if __name__ == '__main__':
    if not args.only_evaluate:
        main()
    print('now starting evaluate test dataset')
    test_loss, test_acc = evaluate(test_dataset)
    print(f'test done! test_loss {test_loss} test_acc {test_acc}')