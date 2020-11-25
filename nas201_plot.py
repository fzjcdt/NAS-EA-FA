import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '#':
                continue
            line_arr = line.split(',')
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


datasets = ['cifar10-valid', 'cifar100', 'ImageNet16-120']
for dataset in datasets:
    file_type = 'valid'

    plot_result_with_25_75('./result/nas201/' + dataset + '/rs_' + file_type + '.txt', 'random search')
    plot_result_with_25_75('./result/nas201/' + dataset + '/re_' + file_type + '.txt', 'regularized evolution')
    plot_result_with_25_75('./result/nas201/' + dataset + '/rs_improved_' + file_type + '.txt', 'improved random search')
    plot_result_with_25_75('./result/nas201/' + dataset + '/re_improved_' + file_type + '.txt', 'improved regularized evolution')
    plot_result_with_25_75('./result/nas201/' + dataset + '/neural_predictor_' + file_type + '.txt', 'iterative neural predictor')
    plot_result_with_25_75('./result/nas201/' + dataset + '/nas_ea_fa_' + file_type + '.txt', 'fitness approximation')
    plot_result_with_25_75('./result/nas201/' + dataset + '/nas_ea_fa_v2_' + file_type + '.txt', 'fitness approximation v2')


    plt.legend()
    plt.title(dataset + '-' + file_type)
    plt.xlabel('time clock(second)')
    plt.ylabel(file_type + ' accuracy')

    if dataset == 'cifar10-valid':
        plt.ylim((0.895, 0.920))
    elif dataset == 'cifar100':
        plt.ylim((0.69, 0.74))
    else:
        plt.ylim((0.43, 0.48))
    plt.grid()
    plt.show()
