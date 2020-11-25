import numpy as np
from numpy.random import random

INTERVAL = 1000
REPEAT_TIMES = 300
POPULATION_SIZE = 10

ops_type = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']

hash2info = np.load('./model/nas201.npy', allow_pickle=True)[()]
hash2arch = np.load('./model/nas201_hash2arch.npy', allow_pickle=True)[()]
arch2hash = np.load('./model/nas201_arch2hash.npy', allow_pickle=True)[()]
arch2sequence = np.load('./model/nas201_arch2sequence.npy', allow_pickle=True)[()]

datasets = ['cifar10-valid', 'cifar100', 'ImageNet16-120']
dataset2time_budget = {'cifar10-valid': 200000, 'cifar100': 400000, 'ImageNet16-120': 2000000}


class Individual(object):

    def __init__(self):
        self.ops = np.random.choice(ops_type, size=6)
        self.fitness = 0.0

    def get_arch(self) -> str:
        ops = self.ops
        arch = '|' + ops[0] + '~0|+|' + ops[1] + '~0|' + ops[2] + '~1|+|' + ops[3] + '~0|' + ops[4] + '~1|' + ops[
            5] + '~2|'

        return arch


def get_individual_hash(indi: Individual) -> str:
    return arch2hash[indi.get_arch()]


def get_individual_data(indi: Individual, dataset: str):
    metrics = hash2info[dataset][get_individual_hash(indi)]
    test_acc = np.array([m['test_acc'] for m in metrics]).mean() / 100.0
    rand_index = np.random.randint(len(metrics))
    valid_acc = metrics[rand_index]['valid_acc'] / 100.0
    training_time = metrics[rand_index]['training_time']

    return valid_acc, test_acc, training_time


def get_individual_data_from_history(individual: Individual, history2acc: dict, dataset: str):
    individual_hash = get_individual_hash(individual)
    if individual_hash in history2acc.keys():
        acc = history2acc[individual_hash]
        valid_acc, test_acc, time = acc[0], acc[1], 0.0
    else:
        valid_acc, test_acc, time = get_individual_data(individual, dataset)
        history2acc[individual_hash] = (valid_acc, test_acc)

    return valid_acc, test_acc, time


def get_individual_sequences(indiv: Individual) -> list:
    return arch2sequence[indiv.get_arch()]


def get_all_isomorphic_sequences(individual: Individual) -> list:
    visited = set()
    sequences = []
    indiv_hash = get_individual_hash(individual)
    all_arch = hash2arch[indiv_hash]
    for arch in all_arch:
        seq = arch2sequence[arch]
        seq_str = ''.join(map(str, seq))
        if seq_str not in visited:
            sequences.append(seq)
            visited.add(seq_str)

    return sequences


def bitwise_mutation(individual: Individual):
    mutation_rate = 1.0 / (len(individual.ops))
    for pos in range(len(individual.ops)):
        if random() < mutation_rate:
            temp_ops_types = [t for t in ops_type if t != individual.ops[pos]]
            individual.ops[pos] = np.random.choice(temp_ops_types)

