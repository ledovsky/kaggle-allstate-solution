import sys

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid


default_output_path = '../run_res/grid_search_xgb.tsv'


class FeatureExtractor(TransformerMixin):
    def __init__(self):
        pass

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

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def target_transform(y):
    return np.log(y + 200)


def target_inverse_transform(y_tr):
    return np.exp(y_tr) - 200


def mae_eval(y_true, y_pred_tr):
    y_pred = target_inverse_transform(y_pred_tr)
    return mean_absolute_error(y_true, y_pred)


def xgb_eval(y_pred, dmat):
    y_true = dmat.get_label()
    return 'custom_mae', mae_eval(y_true, y_pred)


def get_train_val(X, y, train,  val):
    X_train = X.iloc[train, :]
    y_train = y.loc[train]
    y_train = target_transform(y_train)
    X_val = X.iloc[val, :]
    y_val = y.loc[val]
    return (xgb.DMatrix(X_train, y_train.values),
            xgb.DMatrix(X_val, y_val), y_val)


param_grid = ParameterGrid({
    'num_boost_round': [100, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'eta': [0.01, 0.05, 0.1, 0.5, 1],
    'objective': ['reg:linear'],
    'eval_metric': ['mae'],
    'silent': [1]
})


if __name__ == "__main__":

    # Set up output path
    try:
        output_path = sys.argv[1]
    except:
        output_path = default_output_path

    # Read data
    df = pd.read_csv('../raw_data/train.csv')
    df_test = pd.read_csv('../raw_data/test.csv')

    X = df.drop(['loss'], 1)
    y = df.loss
    X_test = df_test

    # Set up cross validation
    cv = KFold(X.shape[0], n_folds=3)

    # Apply label encoder
    fe = FeatureExtractor().fit(X, X_test)
    X_tr = fe.transform(X)

    # Run grid search
    with open(output_path, 'w') as f:
        for params in param_grid:
            params_copy = dict(params)  # lets make a copy for output
            num_boost_round = params.pop('num_boost_round')
            scores = []
            for train, val in cv:
                dtrain, dval, y_true = get_train_val(X_tr, y, train, val)
                watchlist = [(dval, 'val')]
                xgbm = xgb.train(params, dtrain, num_boost_round,
                                 evals=watchlist, early_stopping_rounds=25,
                                 verbose_eval=False)
                y_pred_tr = xgbm.predict(dval)
                score = mae_eval(y_true, y_pred_tr)
                scores.append(score)

            score = np.mean(scores)
            f.write('MAE = {:.2f}\t params = {}\n'.
                    format(score, params_copy.__str__()))
