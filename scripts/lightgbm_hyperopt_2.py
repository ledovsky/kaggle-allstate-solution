import sys
import os

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials

from bson.json_util import dumps

from lightgbm_2_objective import evaluate_lightgbm

default_output_path = '../run_res/hyperopt_lightgbm_2.json'

os.environ['LIGHTGBM_EXEC'] = '/home/ledovsky/LightGBM/lightgbm'


def main(output_path):

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

    # trials = Trials()
    trials = MongoTrials('mongo://localhost:27017/allstate/jobs', exp_key='lightgbm_2')

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
