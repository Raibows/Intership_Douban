from matplotlib import pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

font = {'family' : 'SimHei',
        'weight' : 'bold',
        'size'   : 6}

plt.rc('font', **font)

def data_plot_show(piedata, bardata, city_money):
    fig = plt.figure(dpi=300)

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    data = [x[1] for x in piedata]
    labels = [x[0] + f"{int(x[1]*sum(data)/100)}" for x in piedata]
    ax1.pie(data, labels=labels, explode=[0, 0.3, 0.1, 0])
    ax1.set_title("position counts by program language")

    x = [i for i in range(len(bardata))]
    y = [i[1] for i in bardata]
    bars = ax2.bar(x, y)
    ax2.set_xlabel("职位类别")
    ax2.set_ylabel("薪资(时/元)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([i[0] for i in bardata])
    ax2.set_title("不同的职位薪资对比")


    x = [i for i in range(len(city_money))]
    y = [x[1] for x in city_money]

    bars = ax3.bar(x, y, width=0.5)
    # RGB:#AAFFCC
    colors = ["#" + str(random.randint(100000, 999999)) for i in range(len(city_money))]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    for a, b in zip(x, y):
        ax3.text(a, b + 0.1, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    ax3.plot(x,y,'bp--')
    ax3.set_xlabel("城市")
    ax3.set_ylabel("薪资(元)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([x[0] for x in city_money], rotation = 90)
    ax3.set_title("不同的城市主要技术领域薪资对比")

    fig.show()




if __name__ == '__main__':
    import logging
    from scrapyspider.ohmy_DB import douban_db

    logger = logging.getLogger('datashow')
    douban_db = douban_db.douban_db('./database.json', logger)
    douban_db.execute()