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
            data.append([float(v) * 100 for v in line_arr])

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


plt.figure(figsize=(6, 6))
file_type = 'valid'
plot_result_with_25_75('./result/nas101/rs_' + file_type + '.txt', 'improved random search')
plot_result_with_25_75('./result/nas101/re_' + file_type + '.txt', 'improved regularized evolution')
plot_result_with_25_75('./result/nas101/neural_predictor_' + file_type + '.txt', 'iterative neural predictor')
plot_result_with_25_75('./result/nas101/nas_ea_fa_' + file_type + '.txt', 'nas-ea-fa')
plot_result_with_25_75('./result/nas101/nas_ea_fa_v2_' + file_type + '.txt', 'nas-ea-fa v2')


plt.legend()
plt.title('nas101-cifar10 validation set')
plt.xlabel('training time(seconds)')
plt.ylabel(file_type + ' accuracy %')
plt.ylim((94.4, 95.3))

plt.grid()
plt.show()



plt.figure(figsize=(6, 6))
file_type = 'test'
plot_result_with_25_75('./result/nas101/rs_' + file_type + '.txt', 'improved random search')
plot_result_with_25_75('./result/nas101/re_' + file_type + '.txt', 'improved regularized evolution')
plot_result_with_25_75('./result/nas101/neural_predictor_' + file_type + '.txt', 'iterative neural predictor')
plot_result_with_25_75('./result/nas101/nas_ea_fa_' + file_type + '.txt', 'nas-ea-fa')
plot_result_with_25_75('./result/nas101/nas_ea_fa_v2_' + file_type + '.txt', 'nas-ea-fa v2')


plt.legend()
plt.title('nas101-cifar10 test set')
plt.xlabel('training time(seconds)')
plt.ylabel(file_type + ' accuracy %')
plt.ylim((93.4, 94.3))

plt.grid()
plt.show()


