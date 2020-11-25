import numpy as np


def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '#':
                continue
            line_arr = line.split(',')
            data.append(float(line_arr[-1]))

    return data


methods = ['rs', 're', 'rs_improved', 're_improved', 'neural_predictor', 'nas_ea_fa', 'nas_ea_fa_v2']
datasets = ['cifar10-valid', 'cifar100', 'ImageNet16-120']

for method in methods:
    for dataset in datasets:
        for file_type in ['valid', 'test']:
            file = './result/nas201/' + dataset + '/' + method + '_' + file_type + '.txt'
            t = np.array(read_data(file))
            print(round(t.mean() * 100, 2), end='±')
            print(round(t.std() * 100, 2), end=' ')
    print()

