import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

from pylightgbm.models import GBMRegressor

from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, hp, Trials

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from bson.json_util import dumps


default_output_path = '../run_res/hyperopt_lightgbm_3.json'

os.environ['LIGHTGBM_EXEC'] = '/home/ledovsky/LightGBM/lightgbm'




def main(output_path):

    # Set up Hyperopt

    def target_transform(y, mu=200):
        return np.log(y + mu)

    def target_inverse_transform(y_tr, mu=200):
        return np.exp(y_tr) - mu


    start_time = datetime.now()

    # Read and preprocess data

    df = pd.read_csv('/home/ledovsky/allstate/run_res/feat_train_2.csv')
    X = df.drop(['loss', 'id'], 1)
    y = df.loss

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2016)
    n_folds = 4
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=2016)

    def evaluate_lightgbm(params):

        print 'new iteration ', datetime.now().strftime('%H:%M')

        model = GBMRegressor(
            num_threads=8,
            num_iterations=5000,
            verbose=False,
            early_stopping_round=25,
            bagging_seed=2016,
            metric='l1',
            learning_rate=0.1,
            max_depth=12,
            num_leaves=int(params['num_leaves']),
            # num_leaves=127,
            # feature_fraction=params['feature_fraction'],
            # bagging_fraction=params['bagging_fraction'],
            feature_fraction=0.7,
            bagging_fraction=0.7,
            min_data_in_leaf=int(params['min_data_in_leaf']),
            max_bin=int(params['max_bin']),
            # lambda_l1=params['lambda_l1'],
            # lambda_l2=params['lambda_l2']
        )


        for val, train in cv.split(X):
            X_train = X.iloc[train].values
            y_train = y.iloc[train].values
            X_val = X.iloc[val].values
            y_val = y.iloc[val].values

            model.fit(
                X_train,
                target_transform(y_train),
                test_data=[(
                    X_val,
                    target_transform(y_val)
                )]
            )
            best_iter = model.best_round
            y_pred = target_inverse_transform(model.predict(X_val))
            y_pred_train = target_inverse_transform(model.predict(X_train))
            mae = mean_absolute_error(y_val, y_pred)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            break

        # best_iter /= float(n_folds)
        # mae /= n_folds
        # mae_train /= n_folds

        run_time = datetime.now() - start_time

        return {
            'loss': mae,
            'mae_train': mae_train,
            'status': STATUS_OK,
            'best_round': best_iter
        }

    space = {
        # 'max_depth': hp.quniform('max_depth', 13, 13, 1),
        'num_leaves': hp.quniform('num_leaves', 101, 501, 50),
        'max_bin': hp.quniform('max_bin', 63, 511, 64),
        # 'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 0.8, 0.1),
        # 'feature_fraction': hp.quniform('feature_fraction', 0.6, 0.8, 0.1),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 501, 50),
        # 'lambda_l1':  hp.loguniform('lambda_l1', -5, 0),
        # 'lambda_l2':  hp.loguniform('lambda_l2', -5, 0),
    }

    trials = Trials()
    # trials = MongoTrials('mongo://localhost:27017/allstate/jobs', exp_key='lightgbm_2')

    # Run optimization

    fmin(
        fn=evaluate_lightgbm,
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
