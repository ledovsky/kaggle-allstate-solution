import sys

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from bson.json_util import dumps

default_output_path = '../run_res/hyperopt_xgb_3.json'


class FeatureExtractor(TransformerMixin):

    def fit(self, df_1, df_2):
        df = pd.concat([df_1, df_2], axis=0)
        self.cat_columns = [col for col in df.columns if col[:3] == 'cat']
        self.le_dict = {}

        for col in self.cat_columns:
            self.le_dict[col] = LabelEncoder().fit(df[col])

        return self

    def transform(self, df):
        df = df.copy()

        df.drop(['id'], 1, inplace=True)

        for col in self.cat_columns:
            df[col] = self.le_dict[col].transform(df[col])
        return df


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

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.2, random_state=100)
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
        'max_depth': hp.quniform('max_depth', 3, 18, 1),
        'eta': hp.quniform('eta', 0.002, 0.5, 0.002),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.01),
        'gamma':  hp.loguniform('gamma', -5, 2),
        'alpha':  hp.loguniform('alpha', -5, 2),
        'seed': 100
    }

    def evaluate_xgb(params):
        print 'new iteration'
        num_boost_round = 5000
        params = dict(params.items() + base_params.items())
        params['max_depth'] = int(params['max_depth'])

        xgbm = xgb.train(
            params, dtrain, num_boost_round,
            evals=watchlist, early_stopping_rounds=25,
            eval_metric='mae', verbose_eval=False)
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
        max_evals=70,
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
