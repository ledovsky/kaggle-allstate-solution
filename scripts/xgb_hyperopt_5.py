import sys

from datetime import datetime

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from bson.json_util import dumps

default_output_path = '../run_res/hyperopt_xgb_5.json'


def target_transform(y, mu=200):
    return np.log(y + mu)


def target_inverse_transform(y_tr, mu=200):
    return np.exp(y_tr) - mu


def main(output_path):

    # Read and preprocess data

    df = pd.read_csv('../raw_data/train.csv')
    df_test = pd.read_csv('../raw_data/test.csv')
    X = df.drop(['loss'], 1)
    y = df.loss
    X_test = df_test

    fe = FeatureExtractor().fit(X, X_test)
    X_tr = fe.transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.2, random_state=2016)
    dtrain = xgb.DMatrix(X_train, target_transform(y_train))
    dval = xgb.DMatrix(X_val, target_transform(y_val))
    watchlist = [(dval, 'val')]

    base_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'nthread': 6
    }

    # Set up Hyperopt

    space = {
        'max_depth': hp.quniform('max_depth', 9, 15, 1),
        'subsample': hp.quniform('subsample', 0.6, 0.9, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.7, 0.01),
        'gamma':  hp.lognormal('gamma', 0, 1),
        'alpha':  hp.lognormal('alpha', 0, 1),
        'min_child_weight': 1,
        'eta': 0.01,
        'seed': 2016
    }

    def evaluate_xgb(params):
        print 'new iteration ', datetime.now().strftime('%H:%M')
        num_boost_round = 5000
        params = dict(params.items() + base_params.items())
        params['max_depth'] = int(params['max_depth'])

        xgbm = xgb.train(
            params, dtrain, num_boost_round,
            evals=watchlist, early_stopping_rounds=25,
            verbose_eval=False)
        best_iter = xgbm.best_iteration
        y_pred = target_inverse_transform(xgbm.predict(dval))
        mae = mean_absolute_error(y_val, y_pred)

        return {
            'loss': mae,
            'status': STATUS_OK,
            'best_round': best_iter
        }

    trials = Trials()

    # Run optimization

    fmin(
        fn=evaluate_xgb,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
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
