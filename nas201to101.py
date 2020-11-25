import re
import copy as cp
import hashlib

import numpy as np
import torch

from nasbench.lib.my_model_spec import ModelSpec


file_path = 'C:/Users/86131/Downloads/NAS-Bench-201-v1_1-096897.pth'
INPUT = 'input'
OUTPUT = 'output'
ops_type = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
ops2one_hot = {'nor_conv_3x3': [0, 0, 0, 1], 'nor_conv_1x1': [0, 0, 1, 0],
               'avg_pool_3x3': [0, 1, 0, 0], 'skip_connect': [1, 0, 0, 0], 'none': [0, 0, 0, 0]}

base_matrix = np.array([
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
])


def arch2hash_and_sequence(arch: str):
    temp_matrix = cp.deepcopy(base_matrix)
    nodes = re.split('[+|]', arch)
    index = 0
    ops = [INPUT]
    for node in nodes:
        if node != '':
            index += 1
            ops.append(node[:-2])
            if node[:-2] == 'none':
                for col in range(8):
                    temp_matrix[index][col] = 0

    ops.append(OUTPUT)
    model_spec = ModelSpec(temp_matrix, ops)
    if model_spec.matrix is None:
        model_hash = hashlib.md5('invalid'.encode('utf-8')).hexdigest()
        sequence = [0 for _ in range(24)]
    else:
        model_hash = model_spec.hash_spec(ops_type)
        sequence = []
        for i in range(1, len(ops) - 1):
            if i in model_spec.extraneous:
                sequence.extend(ops2one_hot['none'])
            else:
                sequence.extend(ops2one_hot[ops[i]])

    return model_hash, sequence


def update_dataset2info(dataset: str, model_hash: str, data: dict):
    if model_hash in dataset2info[dataset].keys():
        dataset2info[dataset][model_hash].append(data)
    else:
        dataset2info[dataset][model_hash] = [data]


def update_hash2arch(hash: str, arch: str):
    if hash in hash2arch.keys():
        hash2arch[hash].append(arch)
    else:
        hash2arch[hash] = [arch]


dataset2info = {
    'cifar10': {},
    'cifar10-valid': {},
    'cifar100': {},
    'ImageNet16-120': {}
}

hash2arch, arch2hash, arch2sequence = {}, {}, {}

file = torch.load(file_path)
for xkey in sorted(list(file['arch2infos'].keys())):
    all_info = file['arch2infos'][xkey]['full']
    arch_str = all_info['arch_str']
    model_hash, sequence = arch2hash_and_sequence(arch_str)
    update_hash2arch(model_hash, arch_str)
    arch2hash[arch_str] = model_hash
    arch2sequence[arch_str] = sequence
    for dataset, metric in all_info['all_results'].items():
        data = {
            'training_acc': metric['train_acc1es'][199],
            'test_acc': metric['eval_acc1es']['ori-test@199'],
            'training_time': sum(metric['train_times'].values())
        }
        if dataset[0] != 'cifar10':
            data['valid_acc'] = metric['eval_acc1es']['x-valid@199']

        update_dataset2info(dataset[0], model_hash, data)


np.save('./model/nas201_.npy', dataset2info)
np.save('./model/nas201_hash2arch_.npy', hash2arch)
np.save('./model/nas201_arch2hash_.npy', arch2hash)
np.save('./model/nas201_arch2sequence_.npy', arch2sequence)

"""
cifar10, cifar10-valid, cifar100, ImageNet16-120

{model_hash: [
    {training_acc: xxx, valid_acc: xx, test_acc: xx, training_time: xxx}, 
    {training_acc: xxx, valid_acc: xx, test_acc: xx, training_time: xxx},
    {training_acc: xxx, valid_acc: xx, test_acc: xx, training_time: xxx}], 
    [{training_acc: xxx, valid_acc: xx, test_acc: xx, training_time: xxx}, 
    {training_acc: xxx, valid_acc: xx, test_acc: xx, training_time: xxx},
    {training_acc: xxx, valid_acc: xx, test_acc: xx, training_time: xxx}]
}

{hash1: [arch1, arch2], hash2: [arch3, arch4]}
"""

