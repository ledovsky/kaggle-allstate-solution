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

    fe = FeatureExtractor().fit(df, df_test)
    df_tr = fe.transform(df)

    X_test = fe.transform(df_test)
    dtest = xgb.DMatrix(X_test)
    y_pred = np.zeros(X_test.shape[0])

    params = {
        'alpha': 2.8057319601765127,
        'colsample_bytree': 0.46,
        'max_depth': 13,
        'gamma': 0.9945292474298767,
        'subsample': 0.9,
        'eta': 0.01,
        'seed': 2016,
        'min_child_weight': 1,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }

    num_boost_round = 2267

    random_state_base = 2016

    cv = KFold(n_splits=10)

    for i in range(n_bags):
        print 'iter', i

        random_state = random_state_base * (i + 1)

        sub_df = df_tr.sample(frac=0.8, random_state=random_state)

        X_train = sub_df.drop(['loss'], 1)
        y_train = sub_df.loss
        dtrain = xgb.DMatrix(X_train, target_transform(y_train))

        params['seed'] = random_state

        xgbm = xgb.train(params, dtrain, num_boost_round)
        y_pred += target_inverse_transform(xgbm.predict(dtest))

    y_pred /= n_bags
    submission = get_submission(y_pred)
    submission.to_csv('../submissions/11_24_1.csv', index=False)
