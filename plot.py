import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_arr = line.strip().split(',')
            data.append([float(v) for v in line_arr])

    return data


def plot_result(file_name, label=None):
    data = np.array(read_data(file_name))
    data = data.mean(axis=0)
    plt.plot([1000 * i for i in range(len(data))], data, label=label)


def plot_result_with_25_75(file_name, label=None):
    data = np.array(read_data(file_name))
    data.sort(axis=0)

    data_mean = data.mean(axis=0)
    data_25 = data[len(data) // 4]
    data_75 = data[len(data) // 4 * 3]

    plt.plot([1000 * i for i in range(len(data_mean))], data_mean, label=label)
    plt.fill_between([1000 * i for i in range(len(data_25))], data_25, data_75, alpha=0.1, linewidth=0)


plot_result_with_25_75('./result/random_search_valid.txt', 'valid')
plot_result_with_25_75('./result/random_search_test.txt', 'test')
plt.legend()
plt.ylim((0.92, 0.96))
plt.show()
