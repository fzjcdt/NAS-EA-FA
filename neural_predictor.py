import numpy as np
from numpy.random import randint
from sklearn.svm import SVR

from nasbench import api

# NASBENCH_TFRECORD = './model/nasbench_full.tfrecord'
NASBENCH_TFRECORD = './model/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

MAX_TIME_BUDGET = 1000000
EPOCH_TIME = 250000
INTERVAL = 1000
REPEAT_TIMES = 1000
VALID_RESULT_FILE = './result/neural_predictor_valid.txt'
TEST_RESULT_FILE = './result/neural_predictor_test.txt'

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


def get_model_sequences(model: Model) -> list:
    return get_sequences(model.spec.ops, model.spec.matrix)


def get_model_acc(model: Model):
    _, computed_stat = nasbench.get_metrics_from_spec(model.spec)
    valid_acc = computed_stat[108][randint(3)]['final_validation_accuracy']
    test_acc = np.array([computed_stat[108][i]['final_test_accuracy'] for i in range(3)]).mean()

    return valid_acc, test_acc


def get_model_data(model: Model):
    _, computed_stat = nasbench.get_metrics_from_spec(model.spec)
    rand_index = randint(3)
    valid_acc = computed_stat[108][rand_index]['final_validation_accuracy']
    test_acc = np.array([computed_stat[108][i]['final_test_accuracy'] for i in range(3)]).mean()
    training_time = computed_stat[108][rand_index]['final_training_time']

    return valid_acc, test_acc, training_time


def generate_training_dataset(n=100):
    x_train, y_train, test_acc, time = [], [], [], []
    for i in range(n):
        model = Model()
        x_train.append(get_model_sequences(model))
        # valid_acc, _ = get_model_acc(model)
        v_acc, t_acc, training_time = get_model_data(model)
        y_train.append(v_acc)
        test_acc.append(t_acc)
        time.append(training_time)

    return x_train, y_train, test_acc, time


def get_all_dataset():
    x, y, test_acc, time = [], [], [], []
    for unique_hash in nasbench.hash_iterator():
        fixed_stat, computed_stat = nasbench.get_metrics_from_hash(unique_hash)
        rand_index = randint(3)
        valid_acc = computed_stat[108][rand_index]['final_validation_accuracy']
        t_acc = np.array([computed_stat[108][i]['final_test_accuracy'] for i in range(3)]).mean()
        training_time = computed_stat[108][rand_index]['final_training_time']
        arch = api.ModelSpec(
            matrix=fixed_stat['module_adjacency'],
            ops=fixed_stat['module_operations'],
        )
        x.append(get_sequences(arch.ops, arch.matrix))
        y.append(valid_acc)
        test_acc.append(t_acc)
        time.append(training_time)

    return x, y, test_acc, time


def neural_predictor():
    cur_time_budget = 0
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]
    x_train, y_train, test_acc, training_dataset_time = generate_training_dataset(n=100)
    for v_acc, t_acc, time in zip(y_train, test_acc, training_dataset_time):
        if v_acc > best_valid_acc[-1]:
            best_valid_acc.append(v_acc)
            best_test_acc.append(t_acc)
        else:
            best_valid_acc.append(best_valid_acc[-1])
            best_test_acc.append(best_test_acc[-1])
        times.append(time)
        cur_time_budget += time

    svr = SVR()
    svr.fit(x_train, y_train)

    x, y, test_acc, time = get_all_dataset()
    y_predict = np.array(svr.predict(x))
    sorted_index = (-y_predict).argsort()
    x, y, test_acc, time = np.array(x)[sorted_index], np.array(y)[sorted_index], np.array(test_acc)[sorted_index], \
                           np.array(time)[sorted_index]

    index = 0
    while cur_time_budget < MAX_TIME_BUDGET:
        if y[index] > best_valid_acc[-1]:
            best_valid_acc.append(y[index])
            best_test_acc.append(test_acc[index])
        else:
            best_valid_acc.append(best_valid_acc[-1])
            best_test_acc.append(best_test_acc[-1])

        times.append(time[index])
        cur_time_budget += time[index]
        index += 1

    return best_valid_acc, best_test_acc, times


def run_neural_predictor():
    best_valid_acc, best_test_acc, times = neural_predictor()
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
        valid_acc, test_acc = run_neural_predictor()
        write2file(VALID_RESULT_FILE, valid_acc)
        write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()

