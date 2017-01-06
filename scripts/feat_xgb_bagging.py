from datetime import datetime

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def target_transform(y, mu=200):
    return np.log(y + mu)


def target_inverse_transform(y_tr, mu=200):
    return np.exp(y_tr) - mu


def get_submission(y_sub):
    df_sub = df_test[['id']].copy()
    df_sub['loss'] = y_sub
    return df_sub


def xgb_eval(y_pred_tr, dmat):
    y_pred = target_inverse_transform(y_pred_tr)
    y_true = dmat.get_label()
    return 'custom_mae', mean_absolute_error(y_true, y_pred)

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess


params = {
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'eta': 0.03,
    'max_depth': 12,
    'min_child_weight': 100,
    'seed': 2016,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': True
}

df = pd.read_csv('../run_res/feat_train.csv')
df_test = pd.read_csv('../run_res/feat_test.csv')
X = df.drop(['loss', 'id'], 1)
X_test = df_test.drop(['id'], 1)
y = df.loss

dtest = xgb.DMatrix(X_test.values)


n_folds = 10
cv = KFold(n_splits=n_folds, shuffle=True, random_state=2016)

y_oob = np.zeros(X.shape[0])
y_pred = np.zeros(df_test.shape[0])
cv_score = 0
best_iter = 0

i = 0
for train, val in cv.split(X):
    print 'Iteration', i, datetime.now().strftime('%H:%M')
    i += 1

    X_train = X.iloc[train].values
    y_train = y.iloc[train].values
    X_val = X.iloc[val].values
    y_val = y.iloc[val].values

    dtrain = xgb.DMatrix(X_train, target_transform(y_train))
    dval = xgb.DMatrix(X_val, y_val)
    watchlist = [(dval, 'val')]

    # params['seed'] = random_state

    num_boost_round = 10000

    xgbm = xgb.train(params, dtrain, num_boost_round,
                     evals=watchlist, early_stopping_rounds=50,
                     verbose_eval=True,
                     obj=fair_obj,
                     feval=xgb_eval)

    y_oob[val] = target_inverse_transform(xgbm.predict(dval))
    y_pred += target_inverse_transform(xgbm.predict(dtest))

    mae = mean_absolute_error(y_val, y_oob[val])
    cv_score += mae
    best_iter += xgbm.best_iteration

    print 'MAE = {}, BEST ITER = {}'.format(
        mae,
        xgbm.best_iteration
    )


df_oob = df[['id']].copy()
df_oob['loss'] = y_oob
df_oob.to_csv('../run_res/feat_xgb_bag_oob_2.csv', index=False)

y_pred /= n_folds
submission = df_test[['id']].copy()
submission['loss'] = y_pred
submission.to_csv('../submissions/feat_xgb_bag_2.csv', index=False)

cv_score /= n_folds
print 'Overall CV MAE = {}'.format(cv_score)

best_iter /= float(n_folds)
print 'Average best iteration = {}'.format(best_iter)
