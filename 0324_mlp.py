import numpy as np
import matplotlib.pyplot as plt



np.random.seed(667)


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


class MLP:
    def __init__(self, num_layer, input_size, hidden_size, label_num):
        if num_layer == 1: assert hidden_size == label_num
        self.input_size = input_size
        self.num_layer = num_layer
        self.label_num = label_num
        self.w = [np.random.normal(0, 0.01, [hidden_size, input_size])]
        self.b = [np.random.normal(0, 0.01, [hidden_size, 1])]
        self.w_grad = [np.zeros([hidden_size, input_size])]
        self.b_grad = [np.zeros([hidden_size, 1])]
        for i in range(num_layer-2):
            self.w.append(np.random.normal(0, 0.01, [hidden_size, hidden_size]))
            self.b.append(np.random.normal(0, 0.01, [hidden_size, 1]))
            self.w_grad.append(np.zeros([hidden_size, hidden_size]))
            self.b_grad.append(np.zeros([hidden_size, 1]))
        self.w.append(np.random.normal(0, 0.01, [label_num, hidden_size]))
        self.b.append(np.random.normal(0, 0.01, [label_num, 1]))
        self.w_grad.append(np.zeros([label_num, hidden_size]))
        self.b_grad.append(np.zeros([label_num, 1]))


    def forward(self, x:np.ndarray, lr):
        outs = [x]
        for i in range(self.num_layer):
            temp = self.w[i] @ outs[i].T + self.b[i]
            temp = self.relu_activation(temp)
            outs.append(temp.T)

        for i


    def relu_activation(self, x):
        x[x < 0] = 0
        return x

    def sigmoid_activation(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    model = MLP(num_layer=3, input_size=5, hidden_size=30, label_num=3)
    x = np.random.normal(0, 0.1, [10, 5])
    model.forward(x, 0.1)
