from nas101_utils import Individual
from nas101_utils import MAX_TIME_BUDGET, REPEAT_TIMES, INTERVAL
from nas101_utils import get_model_hash, get_model_data
from utils import write2file, handle_result

VALID_RESULT_FILE = './result/nas101/rs_valid.txt'
TEST_RESULT_FILE = './result/nas101/rs_test.txt'


def random_search():
    cur_time_budget = 0
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]
    history_hash = set()
    while cur_time_budget <= MAX_TIME_BUDGET:
        individual = Individual()
        individual_hash = get_model_hash(individual)
        if individual_hash in history_hash:
            continue
        else:
            history_hash.add(individual_hash)
        valid_acc, test_acc, time = get_model_data(individual)
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
    valid_acc, test_acc = handle_result(best_valid_acc, best_test_acc, times, MAX_TIME_BUDGET, INTERVAL)

    return valid_acc, test_acc


def main():
    for r in range(REPEAT_TIMES):
        print(r)
        valid_acc, test_acc = run_random_search()
        write2file(VALID_RESULT_FILE, valid_acc)
        write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
