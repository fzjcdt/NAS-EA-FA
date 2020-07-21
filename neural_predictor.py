import numpy as np
from numpy.random import randint

from nasbench import api

from sklearn.svm import SVR

# NASBENCH_TFRECORD = './model/nasbench_full.tfrecord'
NASBENCH_TFRECORD = './model/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

ops_type = [CONV1X1, CONV3X3, MAXPOOL3X3]
ops2one_hot = {CONV1X1: [1, 0, 0], CONV3X3: [0, 1, 0], MAXPOOL3X3: [0, 0, 1]}

nasbench = api.NASBench(NASBENCH_TFRECORD)


class Model(object):
    def __init__(self):
        self.ops = self.init_ops()
        self.matrix = self.init_adjacent_matrix()
        while not self.is_valid_connection():
            self.matrix = self.init_adjacent_matrix()

        self.spec = api.ModelSpec(self.matrix, self.ops)

    def is_valid_connection(self):
        return nasbench.is_valid(api.ModelSpec(self.matrix, self.ops))

    @staticmethod
    def init_adjacent_matrix():
        return np.triu(randint(0, 2, size=(7, 7)), k=1)

    @staticmethod
    def init_ops():
        ops = [INPUT] + [ops_type[randint(3)] for _ in range(5)]
        ops.append(OUTPUT)

        return ops

    def get_sequences(self) -> list:
        rst = []
        ops, matrix = self.spec.ops, self.spec.matrix
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


def get_model_acc(model: Model):
    _, computed_stat = nasbench.get_metrics_from_spec(model.spec)
    valid_acc = computed_stat[108][randint(3)]['final_validation_accuracy']
    test_acc = np.array([computed_stat[108][i]['final_test_accuracy'] for i in range(3)]).mean()

    return valid_acc, test_acc


def generate_train_set(n=100):
    x_train, y_train = [], []
    for i in range(n):
        model = Model()
        x_train.append(model.get_sequences())
        valid_acc, _ = get_model_acc(model)
        y_train.append(valid_acc)

    return x_train, y_train


def get_all_dataset():
    x, y = [], []
    for unique_hash in nasbench.hash_iterator():
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
