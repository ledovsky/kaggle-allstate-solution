import sys

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, Trials

from bson.json_util import dumps

default_output_path = '../run_res/hyperopt_xgb.json'


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


def target_transform(y):
    return np.log(y + 200)


def target_inverse_transform(y_tr):
    return np.exp(y_tr) - 200


def main(output_path):

    # Read and preprocess data

    df = pd.read_csv('../raw_data/train.csv')
    df_test = pd.read_csv('../raw_data/test.csv')
    X = df.drop(['loss'], 1)
    y = df.loss
    X_test = df_test

    fe = FeatureExtractor().fit(X, X_test)
    X_tr = fe.transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.33, random_state=42)
    dtrain_1 = xgb.DMatrix(X_train, target_transform(y_train))
    dval_1 = xgb.DMatrix(X_val,  target_transform(y_val))
    y_val_1 = y_val
    watchlist_1 = [(dval_1, 'val')]

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.33, random_state=1042)
    dtrain_2 = xgb.DMatrix(X_train, target_transform(y_train))
    dval_2 = xgb.DMatrix(X_val,  target_transform(y_val))
    y_val_2 = y_val
    watchlist_2 = [(dval_2, 'val')]

    base_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }

    # Set up Hyperopt

    space = {
        'num_boost_round': hp.choice('num_boost_round', [100, 500, 1000]),
        'max_depth': hp.quniform('max_depth', 3, 18, 1),
        'eta': hp.loguniform('eta', np.log(0.001), np.log(0.5)),
        'subsample': hp.uniform('subsample', 0.7, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma':  hp.loguniform('gamma', -5, 1),
    }

    def evaluate_xgb(params):
        num_boost_round = params.pop('num_boost_round')
        params = dict(params.items() + base_params.items())
        params['max_depth'] = int(params['max_depth'])

        # Split 1
        xgbm = xgb.train(params, dtrain_1, num_boost_round,
                         evals=watchlist_1, early_stopping_rounds=25,
                         verbose_eval=False)
        y_pred = target_inverse_transform(xgbm.predict(dval_1))
        mae_1 = mean_absolute_error(y_val_1, y_pred)

        # Split 2
        xgbm = xgb.train(params, dtrain_2, num_boost_round,
                         evals=watchlist_2, early_stopping_rounds=25,
                         verbose_eval=False)
        y_pred = target_inverse_transform(xgbm.predict(dval_2))
        mae_2 = mean_absolute_error(y_val_2, y_pred)
        return (mae_1 + mae_2) / 2

    trials = Trials()

    # Run optimization

    fmin(
        fn=evaluate_xgb,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
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
