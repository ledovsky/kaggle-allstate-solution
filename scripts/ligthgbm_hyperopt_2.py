import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

from pylightgbm.models import GBMRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from bson.json_util import dumps

default_output_path = '../run_res/hyperopt_lightgbm_2.json'

os.environ['LIGHTGBM_EXEC'] = '/home/ledovsky/LightGBM/lightgbm'


def target_transform(y, mu=200):
    return np.log(y + mu)


def target_inverse_transform(y_tr, mu=200):
    return np.exp(y_tr) - mu


def main(output_path):

    # Read and preprocess data

    df = pd.read_csv('../run_res/feat_train.csv')
    X = df.drop(['loss', 'id'], 1)
    y = df.loss

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2016)

    # Set up Hyperopt

    space = {
        'max_depth': hp.quniform('max_depth', 13, 13, 1),
        'num_leaves': hp.quniform('num_leaves', 50, 500, 50),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 0.9, 0.05),
        'feature_fraction': hp.quniform('feature_fraction', 0.25, 0.55, 0.05),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 100, 500, 50),
        'lambda_l1':  hp.loguniform('lambda_l1', -3, 2),
        'lambda_l2':  hp.loguniform('lambda_l2', -3, 2),
    }

    def evaluate_lightgbm(params):
        print 'new iteration ', datetime.now().strftime('%H:%M')

        model = GBMRegressor(
            num_threads=8,
            num_iterations=5000,
            verbose=False,
            early_stopping_round=25,
            bagging_seed=2016,
            metric='l1',
            learning_rate=0.05,
            max_depth=int(params['max_depth']),
            num_leaves=int(params['num_leaves']),
            feature_fraction=params['feature_fraction'],
            bagging_fraction=params['bagging_fraction'],
            min_data_in_leaf=int(params['min_data_in_leaf']),
            lambda_l1=params['lambda_l1'],
            lambda_l2=params['lambda_l2']
        )

        model.fit(
            X_train.values,
            target_transform(y_train.values),
            test_data=[(
                X_val.values,
                target_transform(y_val.values)
            )]
        )
        best_iter = model.best_round
        y_pred = target_inverse_transform(model.predict(X_val))
        y_pred_train = target_inverse_transform(model.predict(X_train))
        mae = mean_absolute_error(y_val, y_pred)
        mae_train = mean_absolute_error(y_val, y_pred_train)

        return {
            'loss': mae,
            'mae_train': mae_train,
            'status': STATUS_OK,
            'best_round': best_iter
        }

    trials = Trials()

    # Run optimization

    fmin(
        fn=evaluate_lightgbm,
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )

    # Print output

    result = dumps(trials.trials)
    with open(output_path, 'w') as f:
        f.write(result)


if __name__ == "__main__":

    # Set up output path
    try:
        output_path = sys.argv[1]
    except:
        output_path = default_output_path

    main(output_path)
