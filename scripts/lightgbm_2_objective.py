from datetime import datetime

import pandas as pd
import numpy as np

from pylightgbm.models import GBMRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from hyperopt import STATUS_OK


def evaluate_lightgbm(params):

    def target_transform(y, mu=200):
        return np.log(y + mu)

    def target_inverse_transform(y_tr, mu=200):
        return np.exp(y_tr) - mu

    print 'new iteration ', datetime.now().strftime('%H:%M')

    # Read and preprocess data

    df = pd.read_csv('/home/ledovsky/allstate/run_res/feat_train.csv')
    X = df.drop(['loss', 'id'], 1)
    y = df.loss

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2016)

    model = GBMRegressor(
        num_threads=7,
        num_iterations=5000,
        verbose=False,
        early_stopping_round=25,
        bagging_seed=2016,
        metric='l1',
        learning_rate=0.1,
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
    mae_train = mean_absolute_error(y_train, y_pred_train)

    return {
        'loss': mae,
        'mae_train': mae_train,
        'status': STATUS_OK,
        'best_round': best_iter
    }
