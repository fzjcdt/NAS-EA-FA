import numpy as np
from xgboost import XGBRegressor

from nas201_utils import Individual
from nas201_utils import REPEAT_TIMES, INTERVAL
from nas201_utils import datasets, dataset2time_budget
from nas201_utils import get_individual_data_from_history, get_individual_sequences
from utils import handle_result, write2file


def sample_models(n=100):
    models = [Individual() for _ in range(n)]

    return models


def neural_predictor(dataset: str):
    cur_time_budget = 0
    history2acc = {}
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]

    models = sample_models(n=100)
    x_train, y_train = [], []

    while cur_time_budget < MAX_TIME_BUDGET:
        models = sorted(models, key=lambda x: x.fitness, reverse=True)
        num = 0
        for model in models:
            v_acc, t_acc, time = get_individual_data_from_history(model, history2acc, dataset)
            if time == 0.0:
                continue

            num += 1
            x_train.append(get_individual_sequences(model))
            y_train.append(v_acc)

            if v_acc > best_valid_acc[-1]:
                best_valid_acc.append(v_acc)
                best_test_acc.append(t_acc)
            else:
                best_valid_acc.append(best_valid_acc[-1])
                best_test_acc.append(best_test_acc[-1])

            times.append(time)
            cur_time_budget += time

            if cur_time_budget >= MAX_TIME_BUDGET or num >= 5:
                break

        gbm = XGBRegressor(eta=0.1)
        gbm.fit(np.array(x_train), np.array(y_train))

        models = sample_models(n=1000)
        sequences = [get_individual_sequences(model) for model in models]
        y_pred = gbm.predict(np.array(sequences))

        for model, pred_fitness in zip(models, y_pred):
            model.fitness = pred_fitness

    return best_valid_acc, best_test_acc, times


def run_neural_predictor(dataset: str):
    best_valid_acc, best_test_acc, times = neural_predictor(dataset)
    valid_acc, test_acc = handle_result(best_valid_acc, best_test_acc, times, MAX_TIME_BUDGET, INTERVAL)
    return valid_acc, test_acc


def main():
    for dataset in datasets:
        VALID_RESULT_FILE = './result/nas201/' + dataset + '/neural_predictor_valid.txt'
        TEST_RESULT_FILE = './result/nas201/' + dataset + '/neural_predictor_test.txt'
        global MAX_TIME_BUDGET
        MAX_TIME_BUDGET = dataset2time_budget[dataset]
        for r in range(REPEAT_TIMES):
            print(r)
            valid_acc, test_acc = run_neural_predictor(dataset)
            write2file(VALID_RESULT_FILE, valid_acc)
            write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
