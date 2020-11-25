import copy as cp

from nas101_utils import Individual
from nas101_utils import MAX_TIME_BUDGET, REPEAT_TIMES, INTERVAL, POPULATION_SIZE
from nas101_utils import get_model_data_from_history, bitwise_mutation
from utils import write2file, handle_result, tournament_selection

VALID_RESULT_FILE = './result/nas101/re_valid.txt'
TEST_RESULT_FILE = './result/nas101/re_test.txt'


def regularized_evolution_algorithm():
    cur_time_budget = 0
    history2acc = {}
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]
    population = [Individual() for _ in range(POPULATION_SIZE)]
    for individual in population:
        valid_acc, test_acc, time = get_model_data_from_history(individual, history2acc)

        individual.fitness = valid_acc
        if valid_acc > best_valid_acc[-1]:
            best_valid_acc.append(valid_acc)
            best_test_acc.append(test_acc)
        else:
            best_valid_acc.append(best_valid_acc[-1])
            best_test_acc.append(best_test_acc[-1])
        times.append(time)
        cur_time_budget += time

    while cur_time_budget <= MAX_TIME_BUDGET:
        individual = cp.deepcopy(tournament_selection(population))
        bitwise_mutation(individual)
        valid_acc, test_acc, time = get_model_data_from_history(individual, history2acc)

        individual.fitness = valid_acc
        if valid_acc > best_valid_acc[-1]:
            best_valid_acc.append(valid_acc)
            best_test_acc.append(test_acc)
        else:
            best_valid_acc.append(best_valid_acc[-1])
            best_test_acc.append(best_test_acc[-1])
        times.append(time)
        population.append(individual)
        population.pop(0)
        cur_time_budget += time
        if cur_time_budget > MAX_TIME_BUDGET:
            break

    return best_valid_acc, best_test_acc, times


def run_evolution_algorithm():
    best_valid_acc, best_test_acc, times = regularized_evolution_algorithm()
    valid_acc, test_acc = handle_result(best_valid_acc, best_test_acc, times, MAX_TIME_BUDGET, INTERVAL)

    return valid_acc, test_acc


def main():
    for r in range(REPEAT_TIMES):
        print(r)
        valid_acc, test_acc = run_evolution_algorithm()
        write2file(VALID_RESULT_FILE, valid_acc)
        write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
