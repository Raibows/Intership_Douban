import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def process_file(path):
    titles = []
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            # line = line.strip('\n').strip(')').strip('(').split(',')
            # print(line)
            line = line.strip('\n').strip(')').strip('(').split(',')
            # print(line)
            if i == 0:
                datas = [[] for _ in line]
                titles.append(line)
                # print(titles)
            else:
                for j, t in enumerate(line):
                    if j == 0: t = int(t)
                    else: t = float(t)
                    datas[j].append(t)

    return titles, datas


def show_model_training_process(datas, titles):
    # 设置图片大小
    # plt.figure(figsize=(15, 20), dpi=300)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 画图——折线图
    plt.plot(datas[0], datas[1], label='train_loss', color="y", marker='o')
    plt.plot(datas[0], datas[2],  label='valid_loss', color="b", marker='1')
    plt.plot(datas[0], datas[3],  label='test_loss', color="r", marker='.')
    plt.plot(datas[0], datas[4],  label='test_acc', color="g", marker='*')
    plt.plot(datas[0], datas[5],  label='test_acc', color="deeppink", marker='d')

    # 设置网格线
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.title("result")
    my_y_ticks = np.arange(0, 1, 0.01)

    plt.yticks(my_y_ticks)
    plt.xlabel('epoch')
    plt.ylabel('0-0.6范围')
    plt.ylim(0, 1)
    x_ticks = np.linspace(0, 1, 10)  # 产生区间在-5至4间的10个均匀数值
    plt.yticks(x_ticks)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
    ax = plt.gca()

    # yminorFormatter = FormatStrFormatter('%0.2f')
    # ax.yaxis.set_minor_formatter(yminorFormatter)
    # yminorLocator = MultipleLocator(0.1)
    # ax.yaxis.set_minor_locator(yminorLocator)

    plt.show()


def setFont():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)  # 设置子图间距

    # 设置字体
    mpl.rcParams['figure.figsize'] = [3.0, 3.0]
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300

    mpl.rcParams['font.size'] = 2
    mpl.rcParams['legend.fontsize'] = 2
    mpl.rcParams['figure.titlesize'] = 2


def getGeners(douban_data):
    ser1 = douban_data['genres']
    for i in range(len(ser1)):
        if ser1[i] is not None and ser1[i] != 'NULL':
            ser1[i] = ser1[i].split("/")
    G_set = set()
    for list1 in ser1:
        if list1 is not None and str(list1) != 'NULL':
            G_set.update(list1)
    G_set = list(G_set)
    no = [0 for i in range(len(G_set))]
    for i in range(len(G_set)):
        for list1 in ser1:
            if list1 is not None and G_set[i] in list1:
                no[i] += 1
    dict1 = dict()
    for k, v in zip(G_set, no):
        dict1[k] = v
    array1 = pd.Series(dict1).sort_values(ascending=False)
    movie_no = list()
    movie_type = list()
    for key in array1.index:
        if len(movie_type) < 15:
            movie_no.append(array1[key])
            movie_type.append(key)
    return movie_type, movie_no


def getRegion(douban_data):
    ser1 = douban_data['region']
    for i in range(len(ser1)):
        if ser1[i] is not None and ser1[i] != 'NULL':
            ser1[i] = ser1[i].split("/")
    G_set = set()
    for list1 in ser1:
        if list1 is not None and str(list1) != 'NULL':
            G_set.update(list1)
    G_set = list(G_set)
    no = [0 for i in range(len(G_set))]
    for i in range(len(G_set)):
        for list1 in ser1:
            if list1 is not None and G_set[i] in list1:
                no[i] += 1
    dict1 = dict()
    for k, v in zip(G_set, no):
        dict1[k] = v
    array1 = pd.Series(dict1).sort_values(ascending=False)
    movie_no = list()
    movie_region = list()
    for key in array1.index:
        if len(movie_region) < 20:
            movie_no.append(array1[key])
            movie_region.append(key)
    return movie_region, movie_no


def getRate(douban_data):
    ser1 = douban_data['rate']
    x2 = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]
    y2 = [0 for i in range(len(x2))]
    for i in ser1:
        if i is not None:
            if 0 <= float(i) < 2:
                y2[0] += 1
            elif float(i) < 4:
                y2[1] += 1
            elif float(i) < 6:
                y2[2] += 1
            elif float(i) < 8:
                y2[3] += 1
            elif float(i) < 10:
                y2[4] += 1
    return x2, y2


def getDirector(douban_data):
    arry1 = douban_data.director.value_counts().sort_values(ascending=False)
    movie_no3 = list()
    director_name = list()
    for i in arry1.index:
        if len(director_name) < 15 and 'NULL' != i:
            director_name.append(i)
            movie_no3.append(arry1[i])
    return director_name, movie_no3


def data_plot_show(ax1data, ax2data, ax3data, ax4data):
    fig = plt.figure(dpi=300, figsize=(4, 4))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    colors = ['#E66B1A', '#F8D8C2', '#E4E492', '#C5ED89', '#8EE98E', '#28C728',
              '#28C7C7', '#188AD7', '#BB98EF', '#E7A0E7', '#EB7ABE', '#CF415E',
              '#FF5555', '#F05731', '#CAEB48', '#B0D718', '#60A0B0', '#5555DD',
              '#8D3CC4', '#BA63D0']

    ax1.bar(ax1data[0], ax1data[1], width=0.8)
    bars1 = ax1.bar(ax1data[0], ax1data[1], width=0.8)

    for bar1, color in zip(bars1, colors):
        bar1.set_color(color)
    # 设置标签
    ax1.set_xlabel('国家')
    ax1.set_ylabel('数量（部）')
    ax1.set_xticks(ax1data[0])
    ax1.set_xticklabels(ax1data[0])
    ax1.set_title("不同国家电影部数TOP20")
    ax1.spines['right'].set_linewidth(0.2)
    ax1.spines['top'].set_linewidth(0.2)
    ax1.spines['bottom'].set_linewidth(0.2)  ###设置底部坐标轴的粗细
    ax1.spines['left'].set_linewidth(0.2)  ####设置左边坐标轴的粗细
    ax1.tick_params(axis='both', which='both', length=0)
    # 为每个柱状块添加具体文字数据
    for i1, j1 in zip(ax1data[0], ax1data[1]):
        ax1.text(i1, j1, '{}部'.format(j1), ha='center', va='bottom')

    ax2.bar(ax2data[0], ax2data[1], width=0.8)
    bars2 = ax2.bar(ax2data[0], ax2data[1], width=0.8)
    for bar2, color in zip(bars2, colors):
        bar2.set_color(color)
    # 设置标签
    ax2.set_xlabel('国家')
    ax2.set_ylabel('数量（部）')
    ax2.set_xticks(ax2data[0])
    ax2.set_xticklabels(ax2data[0], rotation=60)
    ax2.set_title("不同国家电影部数TOP20")
    ax2.spines['right'].set_linewidth(0.2)
    ax2.spines['top'].set_linewidth(0.2)
    ax2.spines['bottom'].set_linewidth(0.2)  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(0.2)  ####设置左边坐标轴的粗细
    ax2.tick_params(axis='both', which='both', length=0)
    # 为每个柱状块添加具体文字数据
    for i2, j2 in zip(ax2data[0], ax2data[1]):
        ax2.text(i2, j2, '{}部'.format(j2), ha='center', va='bottom')

    labels = ["[" + str(i3[0]) + "," + str(i3[1]) + ")" for i3 in ax3data[0]]
    explode = [0, 0, 0, 0, 0.1]
    ax3.pie(ax3data[1], labels=labels, explode=explode)
    ax3.set_title("各个电影评分")

    ax4.bar(ax4data[0], ax4data[1], width=0.8)
    bars4 = ax4.bar(ax4data[0], ax4data[1], width=0.8)
    for bar4, color in zip(bars4, colors):
        bar4.set_color(color)
    # 设置标签
    ax4.set_xlabel('导演名')
    ax4.set_ylabel('数量（部）')
    ax4.set_xticks(ax4data[0])
    ax4.set_xticklabels(ax4data[0], rotation=45)
    ax4.set_title("大导演指导电影部数TOP 15")
    ax4.spines['right'].set_linewidth(0.2)
    ax4.spines['top'].set_linewidth(0.2)
    ax4.spines['bottom'].set_linewidth(0.2)  # 设置坐标轴的粗细
    ax4.spines['left'].set_linewidth(0.2)
    ax4.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
    # 为每个柱状块添加具体文字数据
    for i4, j4 in zip(ax4data[0], ax4data[1]):
        ax4.text(i4, j4, '{}部'.format(j4), ha='center', va='bottom')

    plt.show()


def statistic_data(path):
    setFont()
    csv_data = []
    with open(path, 'r', encoding='utf8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        douban_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            csv_data.append(row)
    douban_header = douban_header[0].split('\t')
    datas = list()
    for data in csv_data:
        ss = data[0].replace("\"", '').split('\t')
        datas.append(ss)
    douban_data = pd.DataFrame(datas)
    douban_data = douban_data.drop([10, 11, 12, 13, 14, 15, 16, 17, 18], axis=1)
    douban_data.columns = douban_header

    movie_type, movie_no = getGeners(douban_data)
    ax1data = [movie_type, movie_no]
    movie_region, movie_no1 = getRegion(douban_data)
    ax2data = [movie_region, movie_no1]
    movie_rate, movie_no2 = getRate(douban_data)
    ax3data = [movie_rate, movie_no2]
    director_name, movie_no3 = getDirector(douban_data)
    ax4data = [director_name, movie_no3]
    data_plot_show(ax1data, ax2data, ax3data, ax4data)



    


if __name__ == '__main__':
    path = './static/03_26_08_46_44_0.43995_0.17659_bert_douban.pt.rec'
    statistic_data(path)
    titles, datas = process_file(path)
    show_model_training_process(datas, titles)


