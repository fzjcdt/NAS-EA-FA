import numpy as np


def handle_result(best_valid_acc, best_test_acc, times, max_time_budget=6000000, interval=1000):
    cur_time, times_len = 0.0, len(times)
    valid_acc, test_acc = [], []
    index = -1
    for time_step in range(0, max_time_budget + 1, interval):
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


def get_sequence_distance(s1, s2) -> int:
    rst = 0
    for t1, t2 in zip(s1, s2):
        if t1 != t2:
            rst += 1
    return rst


def get_min_distance(x_train, s):
    min_d = 100000
    for temp_s in x_train:
        min_d = min(min_d, get_sequence_distance(temp_s, s))

    return min_d


def tournament_selection(population: list, percent=0.2):
    k = int(len(population) * percent)
    individual = np.random.choice(population)
    for _ in range(k - 1):
        temp_individual = np.random.choice(population)
        if temp_individual.fitness > individual.fitness:
            individual = temp_individual

    return individual
