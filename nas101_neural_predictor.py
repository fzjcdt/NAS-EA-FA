import numpy as np
from xgboost import XGBRegressor

from nas101_utils import Individual
from nas101_utils import MAX_TIME_BUDGET, REPEAT_TIMES, INTERVAL
from nas101_utils import get_model_data_from_history, get_model_sequences
from utils import write2file, handle_result

VALID_RESULT_FILE = './result/nas101/neural_predictor_valid.txt'
TEST_RESULT_FILE = './result/nas101/neural_predictor_test.txt'


def sample_models(n=100):
    models = [Individual() for _ in range(n)]

    return models


def neural_predictor():
    cur_time_budget = 0
    history2acc = {}
    best_valid_acc, best_test_acc, times = [0.0], [0.0], [0.0]

    models = sample_models(n=1000)
    x_train, y_train = [], []

    while cur_time_budget < MAX_TIME_BUDGET:
        models = sorted(models, key=lambda x: x.fitness, reverse=True)
        num = 0
        for model in models:
            valid_acc, test_acc, time = get_model_data_from_history(model, history2acc)
            if time == 0.0:
                continue

            x_train.append(get_model_sequences(model))
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

            if cur_time_budget >= MAX_TIME_BUDGET or num >= 50:
                break

        if cur_time_budget >= MAX_TIME_BUDGET:
            break

        gbm = XGBRegressor(eta=0.1)
        gbm.fit(np.array(x_train), np.array(y_train))

        models = sample_models(n=10000)
        sequences = [get_model_sequences(model) for model in models]
        y_pred = gbm.predict(np.array(sequences))
        for index in range(len(models)):
            models[index].fitness = y_pred[index]

    return best_valid_acc, best_test_acc, times


def run_neural_predictor():
    best_valid_acc, best_test_acc, times = neural_predictor()
    valid_acc, test_acc = handle_result(best_valid_acc, best_test_acc, times, MAX_TIME_BUDGET, INTERVAL)

    return valid_acc, test_acc


def main():
    for r in range(REPEAT_TIMES):
        print(r)
        valid_acc, test_acc = run_neural_predictor()
        write2file(VALID_RESULT_FILE, valid_acc)
        write2file(TEST_RESULT_FILE, test_acc)


if __name__ == '__main__':
    main()
