import copy as cp
import numpy as np
from numpy.random import randint, random
import itertools

from nasbench import api

NASBENCH_TFRECORD = './model/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

ops_type = [CONV1X1, CONV3X3, MAXPOOL3X3]
ops2one_hot = {CONV1X1: [1, 0, 0], CONV3X3: [0, 1, 0], MAXPOOL3X3: [0, 0, 1]}

MAX_TIME_BUDGET = 6000000
INTERVAL = 1000
REPEAT_TIMES = 300
POPULATION_SIZE = 100


nasbench = api.NASBench(NASBENCH_TFRECORD)


class Individual(object):

    def __init__(self):
        self.ops = self.init_ops()
        self.connection = self.init_connection()
        while not self.is_valid_connection():
            self.connection = self.init_connection()

        self.fitness = 0.0
        self.spec = None
        self.update_spec()

    def is_valid_connection(self):
        return nasbench.is_valid(api.ModelSpec(self.connections_to_matrix(), self.ops))

    @staticmethod
    def init_connection():
        return np.array([randint(2) for _ in range(21)], dtype=int)

    @staticmethod
    def init_ops():
        ops = [INPUT]
        for _ in range(5):
            ops.append(ops_type[randint(3)])
        ops.append(OUTPUT)

        return ops

    def connections_to_matrix(self):
        rst = np.zeros((7, 7), dtype=int)
        index = 0
        for row in range(7):
            for column in range(row + 1, 7):
                rst[row][column] = self.connection[index]
                index += 1

        return rst

    def update_spec(self):
        self.spec = api.ModelSpec(self.connections_to_matrix(), self.ops)


def get_model_data(individual: Individual):
    model_spec = api.ModelSpec(individual.connections_to_matrix(), individual.ops)
    _, computed_stat = nasbench.get_metrics_from_spec(model_spec)

    test_acc = np.array([computed_stat[108][i]['final_test_accuracy'] for i in range(3)]).mean()
    rand_index = randint(3)
    valid_acc = computed_stat[108][rand_index]['final_validation_accuracy']
    training_time = computed_stat[108][rand_index]['final_training_time']

    return valid_acc, test_acc, training_time


def get_model_data_from_history(individual: Individual, history2acc: dict):
    individual_hash = get_model_hash(individual)
    if individual_hash in history2acc.keys():
        acc = history2acc[individual_hash]
        valid_acc, test_acc, time = acc[0], acc[1], 0.0
    else:
        valid_acc, test_acc, time = get_model_data(individual)
        history2acc[individual_hash] = (valid_acc, test_acc)

    return valid_acc, test_acc, time


def get_model_hash(individual: Individual):
    model_spec = api.ModelSpec(individual.connections_to_matrix(), individual.ops)
    return model_spec.hash_spec(ops_type)


def get_sequences(ops, matrix) -> list:
    rst = []
    v_num = len(ops)
    for i in range(1, 6):
        if i < v_num and ops[i] != OUTPUT:
            rst.extend(ops2one_hot[ops[i]])
        else:
            rst.extend([0, 0, 0])

    for row_index in range(6):
        for col_index in range(row_index + 1, 7):
            if row_index < v_num and col_index < v_num:
                rst.append(matrix[row_index][col_index])
            else:
                rst.append(0)

    return rst


def get_model_sequences(individual: Individual) -> list:
    return get_sequences(individual.spec.ops, individual.spec.matrix)


def _label2ops(label):
    ops = []
    for l in label:
        if l == -1:
            ops.append(INPUT)
        elif l == -2:
            ops.append(OUTPUT)
        else:
            ops.append(ops_type[l])

    return ops


def permute_graph(graph, label, permutation):
    """Permutes the graph and labels based on permutation.
    from nasbench.lib.graph_util import permute_graph

    Args:
      graph: np.ndarray adjacency matrix.
      label: list of labels of same length as graph dimensions.
      permutation: a permutation list of ints of same length as graph dimensions.

    Returns:
      np.ndarray where vertex permutation[v] is vertex v from the original graph
    """
    # vertex permutation[v] in new graph is vertex v in the old graph
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]
    edge_fn = lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1
    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                 (len(label), len(label)),
                                 dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]
    return new_matrix, new_label


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


def get_all_isomorphic_sequences(individual: Individual):
    sequences = []
    matrix = individual.spec.matrix
    label = [-1] + [ops_type.index(op) for op in individual.spec.ops[1:-1]] + [-2]

    vertices = np.shape(matrix)[0]
    # Note: input and output in our constrained graphs always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(1, vertices - 1)):
        full_perm = [0]
        full_perm.extend(perm)
        full_perm.append(vertices - 1)
        pmatrix, plabel = permute_graph(matrix, label, full_perm)
        pmatrix = pmatrix + 0
        ops = _label2ops(plabel)
        if is_upper_triangular(pmatrix) and nasbench.is_valid(api.ModelSpec(pmatrix, ops)):
            sequences.append(get_sequences(ops, pmatrix))

    return sequences


def bitwise_mutation(individual: Individual):
    ops_mutation_rate = 1.0 / (len(individual.ops) - 2)
    for pos in range(1, len(individual.ops) - 1):
        if random() < ops_mutation_rate:
            temp_ops_types = [t for t in ops_type if t != individual.ops[pos]]
            individual.ops[pos] = np.random.choice(temp_ops_types)

    con_mutation_rate = 1.0 / len(individual.connection)
    temp_connection = cp.deepcopy(individual.connection)
    while True:
        for pos in range(len(individual.connection)):
            if random() < con_mutation_rate:
                individual.connection[pos] = 1 - individual.connection[pos]

        if individual.is_valid_connection():
            break
        else:
            individual.connection = cp.deepcopy(temp_connection)

    individual.update_spec()
