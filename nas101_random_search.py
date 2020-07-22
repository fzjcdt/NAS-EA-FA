import numpy as np
from numpy.random import randint

from nasbench import api

# NASBENCH_TFRECORD = './model/nasbench_full.tfrecord'
NASBENCH_TFRECORD = './model/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

ops_type = [CONV1X1, CONV3X3, MAXPOOL3X3]

MAX_TIME_BUDGET = 1000000
INTERVAL = 1000
REPEAT_TIMES = 1000
VALID_RESULT_FILE = './result/random_search_valid.txt'
TEST_RESULT_FILE = './result/random_search_test.txt'

nasbench = api.NASBench(NASBENCH_TFRECORD)


class Individual(object):

    def __init__(self):
        self.ops = self.init_ops()
        self.connection = self.init_connection()
        while not self.is_valid_connection():
            self.connection = self.init_connection()

        self.fitness = self.test_acc = self.valid_acc = self.training_time = 0.0

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


def get_model_acc(individual: Individual):
    model_spec = api.ModelSpec(individual.connections_to_matrix(), individual.ops)
    _, computed_stat = nasbench.get_metrics_from_spec(model_spec)

    rand_index = randint(3)
    test_acc = np.array([computed_stat[108][i]['final_test_accuracy'] for i in range(3)]).mean()
    computed_stat[108][rand_index]['final_test_accuracy'] = test_acc

    return computed_stat[108][rand_index]


def random_search():
    cur_time_budget = 0
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]
    while cur_time_budget <= MAX_TIME_BUDGET:
        individual = Individual()
        data = get_model_acc(individual)
        valid_acc, test_acc, time = data['final_validation_accuracy'], data['final_test_accuracy'], data[
            'final_training_time']
        if valid_acc > best_valid_acc[-1]:
            best_valid_acc.append(valid_acc)
            best_test_acc.append(test_acc)
        else:
            best_valid_acc.append(best_valid_acc[-1])
            best_test_acc.append(best_test_acc[-1])
        times.append(time)
        cur_time_budget += time

    return best_valid_acc, best_test_acc, times


def run_random_search():
    best_valid_acc, best_test_acc, times = random_search()
    cur_time, times_len = 0.0, len(times)
    valid_acc, test_acc = [], []
    index = -1
    for time_step in range(0, MAX_TIME_BUDGET + 1, INTERVAL):
        while cur_time <= time_step:
            index += 1
            if index == times_len:
                break
            cur_time += times[index]
        valid_acc.append(best_valid_acc[index - 1])
        test_acc.append(best_test_acc[index - 1])

    return valid_acc, test_acc


def write2file(file_name: str, data_list: list, clear=False):
    if clear:
        with open(file_name, 'w') as file:
            file.write('')

    data = ','.join(map(str, data_list))
    with open(file_name, 'a') as file:
        file.write(data + '\n')


def main():
    for r in range(REPEAT_TIMES):
        print(r)
        valid_acc, test_acc = run_random_search()
        write2file(VALID_RESULT_FILE, valid_acc)
        write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
