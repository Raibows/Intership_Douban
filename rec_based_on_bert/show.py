import pandas as pd
import matplotlib.pyplot as plt

def process_file(path):
    titles = []
    datas = []
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            line = line.strip('\n').strip(')').strip('(').split(',')
            print(line)
            if i == 0:
                titles.append(line)
            else:
                datas.append(line)
    return titles, datas


def show_model_training_process(df):
    df.head()
    # 设置图片大小
    plt.figure(figsize=(15, 8), dpi=80)
    # 画图——折线图
    plt.plot(df['epoch'], df['train_loss'], label='train_loss', color="y", linestyle='--', linewidth=2)
    plt.plot(df['epoch'], df['valid_loss'], label='valid_loss', color="b", linestyle='--', linewidth=2)
    plt.plot(df['epoch'], df['test_loss'], label='test_loss', color="r", linestyle='--', linewidth=2)
    plt.plot(df['epoch'], df['valid_acc'], label='test_acc', color="g")
    plt.plot(df['epoch'], df['test_acc'], label='test_acc', color="deeppink")

    # 设置网格线
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.title("result")
    plt.show()


if __name__ == '__main__':
    path = './static/03_26_08_46_44_0.43995_0.17659_bert_douban.pt.rec'
    titles, datas = process_file(path)
    df = pd.DataFrame(datas, columns=titles)
    show_model_training_process(df)
