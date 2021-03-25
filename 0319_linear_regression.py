import numpy as np
import matplotlib.pyplot as plt
np.random.seed(667)

def get_fake_data(w_true:list, b_true, num_data, is_noise=True):
    x = np.random.uniform(0, 1, [num_data, len(w_true)])
    w_true = np.array(w_true, dtype=np.float32)
    y = x @ w_true.T + b_true
    if is_noise:
        y += np.random.normal(0, 0.1)
    return x, y

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


class LinearRegression:
    def __init__(self, init_w:list, init_b:list):
        assert len(init_b) == 1
        self.w = np.array(init_w, dtype=np.float32)
        self.w_grad = np.zeros_like(self.w)
        self.b = np.array(init_b, dtype=np.float32)
        self.b_grad = np.zeros_like(self.b)

    def mse_loss(self, p_y:np.ndarray, y:np.ndarray):
        # mean squared error
        return np.sum((p_y - y) ** 2) / 2 / y.shape[0]

    def forward(self, x:np.ndarray, y:np.ndarray):
        p_y = self.predict(x)
        loss = self.mse_loss(p_y, y)
        self.w_grad += ((p_y - y) @ x) / y.shape[0]
        self.b_grad += (p_y - y) / y.shape[0]
        return loss.item()

    def backward(self, lr:float):
        self.w -= self.w_grad * lr
        self.b -= self.b_grad * lr
        self.w_grad.fill(0.0)
        self.b_grad.fill(0.0)

    def predict(self, x:np.ndarray):
        return self.w @ x.T + self.b

def show(x, y, z_true, z_predict):
    figure = plt.figure(1)
    ax = figure.add_subplot(projection='3d')
    ax.scatter(x, y, z_true, marker='o', c='red')
    ax.scatter(x, y, z_predict, marker='^', c='blue')
    plt.show()

if __name__ == '__main__':
    data_x, data_y = get_fake_data([1, 2], 3, 10, is_noise=True)
    epoch = 10
    lr = 0.1
    lr_decay = 0.95
    lr_decay_interval = max(epoch // 5, 1)
    batch_size = max(data_y.shape[0] // 100, 1)

    model = LinearRegression(init_w=[0.0, 0.0], init_b=[0.0])

    for ep in range(epoch):
        loss = 0.0
        for (x, y) in get_batch(data_x, data_y, batch_size):
            loss += model.forward(x, y)
            model.backward(lr)
        if (ep + 1) % lr_decay_interval == 0:
            lr *= lr_decay
            print(f'now lr will step to {lr}')

        w_format = [float(f'{w:.5f}') for w in model.w.tolist()]
        print(f'{ep} epoch done! loss {loss:.5f} w: {w_format} b: {model.b.item():.5f}')

    predicts = model.predict(data_x)
    show(data_x[:, 0], data_x[:, 1], data_y, predicts)

