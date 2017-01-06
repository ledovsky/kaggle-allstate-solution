import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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


def get_submission(y_sub):
    df_sub = df_test[['id']].copy()
    df_sub['loss'] = y_sub
    return df_sub


if __name__ == "__main__":

    # Read and preprocess data

    df = pd.read_csv('../raw_data/train.csv')
    df_test = pd.read_csv('../raw_data/test.csv')
    X = df.drop(['loss'], 1)
    y = df.loss
    X_test = df_test

    fe = FeatureExtractor().fit(X, X_test)
    X_tr = fe.transform(X)
    X_test = fe.transform(df_test)

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.2, random_state=2016)
    dtrain = xgb.DMatrix(X_train, target_transform(y_train))
    dtrain_full = xgb.DMatrix(X_tr, target_transform(y))
    dval = xgb.DMatrix(X_val, target_transform(y_val))
    dtest = xgb.DMatrix(X_test)
    watchlist = [(dval, 'val')]

    params = {
        'alpha': 2.8057319601765127,
        'colsample_bytree': 0.46,
        'max_depth': 13,
        'gamma': 0.9945292474298767,
        'subsample': 0.9,
        'eta': 0.001,
        'seed': 2016,
        'min_child_weight': 1,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'nthread': 4
    }

    num_boost_round = 100000

    xgbm = xgb.train(
        params, dtrain, num_boost_round,
        evals=watchlist, early_stopping_rounds=50,
        verbose_eval=False)
    best_iter = xgbm.best_iteration

    y_pred = target_inverse_transform(xgbm.predict(dval))
    mae = mean_absolute_error(y_val, y_pred)
    print 'MAE = {:.2f}'.format(mae)

    num_boost_round = best_iter
    xgbm = xgb.train(params, dtrain_full, num_boost_round)

    y_pred = target_inverse_transform(xgbm.predict(dtest))

    submission = get_submission(y_pred)
    submission.to_csv('../submissions/11_25_1.csv', index=False)
