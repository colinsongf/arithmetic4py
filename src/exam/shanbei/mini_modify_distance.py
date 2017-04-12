import re
import collections
from collections import Counter
import numpy as np

global letters
global paper
global state
global N
letters = 'abcdefghijklmnopqrstuvwxyz'
paper = ''
state = -1
N = 0


def load(path):
    """
    加载文件，读取语料
    :param path: 文件路径
    :return: 
    """
    with open(path) as file_obj:
        str = file_obj.read()
        str = str.lower()
        str = re.sub('[^A-Za-z0-9]+', ' ', str)
        str = str.strip()
    return str


def compare(source=[], target=[]):
    """
    求交集
    :param source: 
    :param target: 
    :return: 
    """
    interection = list(set(source) & set(target))
    if len(interection) > 0:
        return True
    else:
        return False


def words(text):
    """
    字符串，去噪
    :param text: 
    :return: 
    """
    # text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return re.findall(r'\w+', text.lower())


def save_file(name, l):
    """
    保存结果为文件
    :param name: 文件明
    :param l: 结果集合
    :return: 
    """
    f = open(name, "w")
    for ll in l:
        f.write(ll + "\n")
    f.flush()
    f.close()


def Predit(word):
    """
     概率预测
    :param word: 目标字符串
    :return: 返回在语料中的出现概率
    """
    global N
    global paper
    return paper[word] / N


def known(words):
    """
    计算是否为语料中的单词
    :param words: 字符串
    :return: 如果是语料中的单词就返回单词，否则返回空
    """
    return set(w for w in words if w in paper)


def candidates(word):
    """
    计算不同编辑状态下得到的字符串结果
    :param word: 字符串
    :return: 编辑后的字符串
    """
    global state
    if known([word]):
        state = 1
    elif known(edits1(word)):
        state = 2
    elif known(edits2(word)):
        state = 3
    else:
        state = 0
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def correction(word):
    """
    计算相似性的高的预测结果
    :param word: 目标单词
    :return:返回相似性的高的预测结果 
    """
    global state
    result = max(candidates(word), key=Predit)
    return result


def edits1(word):
    """
    第一次的编辑
    :param word:目标单词 
    :return: 编辑后的单词
    """
    global letters
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """
    第二次的编辑
    :param word: 目标单词
    :return: 编辑后的单词
    """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def plot_scatter_diagram(which_fig, x, y, x_label='x', y_label='y', title='title', label=None):
    styles = ['k.', 'g.', 'r.', 'c.', 'm.', 'y.', 'b.']
    linestyles = ['-.', '--', 'None', '-', ':']
    import pandas
    import numpy

    import numpy as np
    import matplotlib.pyplot as plt
    stylesMarker = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2']

    stylesColors = ['crimson', 'blue', 'red', 'yellow', 'brown', 'green', 'violet']
    assert len(x) == len(y)
    if label != None:
        assert len(x) == len(label) and len(stylesMarker) >= len(set(label))
    plt.figure(which_fig)
    plt.clf()
    if label == None:
        plt.plot(x, y, styles[0])
    else:
        l = len(label)
        labelSet = set(label)
        k = 0
        for i in labelSet:
            xs = []
            ys = []
            for j in range(l):
                if i == label[j]:
                    xs.append(x[j])
                    ys.append(y[j])
            k = k + 1
            plt.scatter(xs, ys, c=stylesColors[k].strip(), marker=stylesMarker[k], label=i)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left')
    plt.show()


def k_means_cluster(data, n_clusters=2):
    """
    Kmeans聚类，对于数据集进行聚类
    :param data: 数据集
    :param n_clusters: 2分类
    :return: 聚类标签
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    print(kmeans.cluster_centers_)
    return kmeans.labels_


if __name__ == '__main__':
    global paper
    global N
    global state
    filename = "eng.txt"
    import time

    paper = load(filename)
    paper = Counter(words(paper))
    N = sum(paper.values())
    f = "tokens.txt"
    ff = open(f, "r")
    data_word = []
    data = []
    i = 0
    while True:
        word = ff.readline().strip()
        print(str(time.strftime("%Y-%m-%d %H:%M %p", time.localtime())) + " computing at " + str(i) + " " + word)
        if not word:
            break
        value = correction(word)
        row = [Predit(word=value), len(word), state]
        data_word.append(word)
        data.append(row)
        i = i + 1
    result_en = []
    result_cn = []
    i = 0
    for ll in data:
        print(str(time.strftime("%Y-%m-%d %H:%M %p", time.localtime())) + " split at " + str(ll[2]) + " " + str(
            ll[1]) + " ->" + str(ll[0]))
        # if ll[1]>=7 or ll[0]>0:
        if ll[0] > 0:
            result_en.append(data_word[i])
        else:
            result_cn.append(data_word[i])
        i = i + 1
    save_file("result_en_a.txt", result_en)
    save_file("result_cn_a.txt", result_cn)
    data = np.array(data)
    print(str(time.strftime("%Y-%m-%d %H:%M %p", time.localtime())) + " finished split job. ")
    labels = k_means_cluster(data, 2)
    print(labels)
    i = 0
    result_en = []
    result_cn = []
    for ii in labels:
        print(str(time.strftime("%Y-%m-%d %H:%M %p", time.localtime())) + " cluster split at " + str(data_word[i]))
        if ii == 0:
            result_en.append(data_word[i])
        else:
            result_cn.append(data_word[i])
        i = i + 1
    save_file("result0_en_b.txt", result_en)
    save_file("result1_cn_b.txt", result_cn)
    plot_scatter_diagram(None, x=data[:, 0], y=data[:, 1], x_label='value', y_label='length',
                         title='k_means value-length', label=labels)
    # the  0.07154004401278254
