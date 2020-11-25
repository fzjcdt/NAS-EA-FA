import copy as cp

import numpy as np
from xgboost import XGBRegressor

from nas101_utils import Individual
from nas101_utils import MAX_TIME_BUDGET, REPEAT_TIMES, INTERVAL, POPULATION_SIZE
from nas101_utils import get_model_data_from_history, bitwise_mutation, get_model_sequences
from utils import write2file, handle_result, tournament_selection

VALID_RESULT_FILE = 'result/nas101/nas_ea_fa_valid.txt'
TEST_RESULT_FILE = 'result/nas101/nas_ea_fa_test.txt'


def NAS_EA_FA():
    cur_time_budget = 0
    history2acc = {}
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]
    x_train, y_train = [], []
    total_population = [Individual() for _ in range(POPULATION_SIZE)]

    while cur_time_budget <= MAX_TIME_BUDGET:
        population = sorted(total_population, key=lambda x: x.fitness, reverse=True)
        new_population = []
        num = 0
        for individual in population:
            valid_acc, test_acc, time = get_model_data_from_history(individual, history2acc)
            individual.fitness = valid_acc
            if time == 0.0:
                continue

            new_population.append(individual)
            x_train.append(get_model_sequences(individual))
            y_train.append(valid_acc)

            if valid_acc > best_valid_acc[-1]:
                best_valid_acc.append(valid_acc)
                best_test_acc.append(test_acc)
            else:
                best_valid_acc.append(best_valid_acc[-1])
                best_test_acc.append(best_test_acc[-1])
            times.append(time)
            cur_time_budget += time
            num += 1
            if cur_time_budget > MAX_TIME_BUDGET or num >= 50:
                break

        if len(new_population) != 0:
            population = new_population

        gbm = XGBRegressor(eta=0.1)
        gbm.fit(np.array(x_train), np.array(y_train))

        total_population = []
        for epoch in range(10):
            new_population = []
            for i in range(POPULATION_SIZE):
                individual = cp.deepcopy(tournament_selection(population))

                bitwise_mutation(individual)
                individual.fitness = gbm.predict(np.array([get_model_sequences(individual)]))[0]
                new_population.append(individual)
                total_population.append(individual)
            population = new_population

    return best_valid_acc, best_test_acc, times


def run_NAS_EA_FA():
    best_valid_acc, best_test_acc, times = NAS_EA_FA()
    valid_acc, test_acc = handle_result(best_valid_acc, best_test_acc, times, MAX_TIME_BUDGET, INTERVAL)

    return valid_acc, test_acc


def main():
    for r in range(REPEAT_TIMES):
        print(r)
        valid_acc, test_acc = run_NAS_EA_FA()
        write2file(VALID_RESULT_FILE, valid_acc)
        write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
