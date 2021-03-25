import numpy as np
import matplotlib.pyplot as plt

# 自适应多分类，自适应多维度，逻辑回归
# 应用交叉熵损失函数

np.random.seed(667)

def get_batch(x, y, batch_size):
    s = 0
    e = s + batch_size
    all = y.shape[0]
    while s < all:
        if e > all:
            yield x[s:], y[s:]
            break
        else:
            yield x[s:e], y[s:e]
            s = e
            e += batch_size

class LogisticRegression:
    def __init__(self, dim, label_num):
        self.w = np.random.normal(0, 0.1, [label_num, dim])
        self.b = np.random.normal(0, 0, [label_num, 1])
        self.w_grad = np.zeros([label_num, dim], dtype=np.float32)
        self.b_grad = np.zeros([label_num, 1], dtype=np.float32)
        self.label_num = label_num

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape([-1, 1])

    def cross_entropy_loss(self, predicts:np.ndarray, y:np.ndarray):
        # do not need log_softmax before input to the cross_entrpy
        # reduction using mean
        temp = -predicts[np.arange(y.shape[0]), y] + np.log(np.sum(np.exp(predicts)))
        return temp[0]

    def predict(self, x):
        return x @ self.w.T + self.b.T

    def forward(self, x, y):
        predicts = self.predict(x)
        loss = self.cross_entropy_loss(predicts, y)
        predicts = self.softmax(predicts).T
        predicts[y, np.arange(y.shape[0])] -= 1
        self.w_grad += predicts @ x / y.shape[0]
        self.b_grad += np.sum(predicts, axis=1).reshape(-1, 1) / y.shape[0]

        return loss.item()

    def backward(self, lr):
        self.w -= lr * self.w_grad
        self.b -= lr * self.b_grad
        self.w_grad.fill(0.0)
        self.b_grad.fill(0.0)

def generate_fake_data(num, dim, label_num):
    # 使用sklearn制造假的分类数据
    from sklearn.datasets import make_classification
    return make_classification(n_samples=num, n_features=dim, n_classes=label_num,
                               n_informative=2, n_redundant=1, n_clusters_per_class=1)

def evaluate(y, predicts):
    acc = (predicts.argmax(1) == y).sum().item() / y.shape[0]
    return acc

def show(loss_records, acc_records):
    plt.plot(loss_records, 'r', label='loss')
    plt.plot(acc_records, 'b', label='accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="center")
    plt.show()


if __name__ == '__main__':
    data_num = 100
    dim = 3
    label_num = 3
    data_x, data_y = generate_fake_data(data_num, dim, label_num)
    epoch = 100
    batch_size = max(data_num // 64, 1)
    lr = 1e-1
    lr_decay = 0.95
    lr_decay_interval = max(epoch // 10, 1)
    early_stop_eps = 1e-4

    model = LogisticRegression(dim, label_num)
    loss_records = []
    acc_records = []

    for ep in range(epoch):
        loss = 0.0
        num = 0
        for (x, y) in get_batch(data_x, data_y, batch_size):
            loss += model.forward(x, y)
            model.backward(lr)
            num += 1
        if (ep + 1) % lr_decay_interval == 0:
            lr *= lr_decay
            print(f'now lr will step to {lr}')
        loss /= num
        loss_records.append(loss)
        predicts = model.predict(data_x)
        acc = evaluate(data_y, predicts)
        acc_records.append(acc)
        print(f'{ep} epoch done! loss {loss:.5f} accuracy {acc:.5f}')
        if loss_records[-1] < early_stop_eps or acc_records[-1] == 1.0 :
            print('the loss is too low, stop early!')
            break

    show(loss_records, acc_records)



