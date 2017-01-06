import sys

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split


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


def get_submission(y_sub):
    df_sub = df_test[['id']].copy()
    df_sub['loss'] = y_sub
    return df_sub


if __name__ == "__main__":

    # Read and preprocess data

    df = pd.read_csv('../run_res/feat_train_2.csv')
    df_test = pd.read_csv('../run_res/feat_test_2.csv')
    x = df.drop(['loss', 'id'], 1)
    x_test = df_test.drop(['id'], 1)
    y = df.loss

    dtest = xgb.dmatrix(x_test.values)


    params = {
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'eta': 0.005,
        'seed': 2016,
        'min_child_weight': 100,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }

    num_boost_round = 5000

    random_state = 2016

    n_folds = 10
    cv = KFold(n_splits=n_folds)

    y_oob = np.zeros(X.shape[0])
    y_pred = np.zeros(df_test.shape[0])

    for train, val in cv.split(X_tr):
        print 'new iter'

        X_train = X_tr.iloc[train].drop(['loss'], 1).values
        y_train = X_tr.iloc[train].loss.values
        X_val = X_tr.iloc[val].drop(['loss'], 1).values
        y_val = X_tr.iloc[val].loss.values

        dtrain = xgb.DMatrix(X_train, target_transform(y_train))
        dval = xgb.DMatrix(X_val, target_transform(y_val))
        watchlist = [(dval, 'val')]

        params['seed'] = random_state

        xgbm = xgb.train(params, dtrain, num_boost_round,
                         evals=watchlist, early_stopping_rounds=25,
                         verbose_eval=False)

        y_oob[val] = target_inverse_transform(xgbm.predict(dval))
        y_pred += target_inverse_transform(xgbm.predict(dtest))

    df_oob = df[['id']].copy()
    df_oob['loss'] = y_oob
    df_oob.to_csv('../run_res/pure_xgb_oob.csv', index=False)

    y_pred /= n_folds
    submission = get_submission(y_pred)
    submission.to_csv('../submissions/11_28_2_xgb_bagging.csv', index=False)
