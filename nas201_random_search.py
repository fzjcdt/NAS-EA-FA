from utils import handle_result, write2file
from nas201_utils import REPEAT_TIMES, INTERVAL
from nas201_utils import Individual
from nas201_utils import get_individual_data
from nas201_utils import datasets, dataset2time_budget


def random_search(dataset: str):
    cur_time_budget = 0
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]
    while cur_time_budget <= MAX_TIME_BUDGET:
        individual = Individual()
        valid_acc, test_acc, time = get_individual_data(individual, dataset)
        if valid_acc > best_valid_acc[-1]:
            best_valid_acc.append(valid_acc)
            best_test_acc.append(test_acc)
        else:
            best_valid_acc.append(best_valid_acc[-1])
            best_test_acc.append(best_test_acc[-1])
        times.append(time)
        cur_time_budget += time

    return best_valid_acc, best_test_acc, times


def run_random_search(dataset: str):
    best_valid_acc, best_test_acc, times = random_search(dataset)
    valid_acc, test_acc = handle_result(best_valid_acc, best_test_acc, times, MAX_TIME_BUDGET, INTERVAL)
    return valid_acc, test_acc


def main():
    for dataset in datasets:
        VALID_RESULT_FILE = './result/nas201/' + dataset + '/rs_valid.txt'
        TEST_RESULT_FILE = './result/nas201/' + dataset + '/rs_test.txt'
        global MAX_TIME_BUDGET
        MAX_TIME_BUDGET = dataset2time_budget[dataset]
        for r in range(REPEAT_TIMES):
            print(r)
            valid_acc, test_acc = run_random_search(dataset)
            write2file(VALID_RESULT_FILE, valid_acc)
            write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
