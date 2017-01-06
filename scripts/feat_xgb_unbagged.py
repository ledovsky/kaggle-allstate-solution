import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.metrics import mean_absolute_error


def target_transform(y, mu=200):
    return (y + mu) ** 0.25


def target_inverse_transform(y_tr, mu=200):
    return (y_tr) ** 0.25 - mu


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
    'eta': 0.005,
    'max_depth': 12,
    'min_child_weight': 100,
    'seed': 2016,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': True
}

df = pd.read_csv('../run_res/feat_train_2.csv')
df_test = pd.read_csv('../run_res/feat_test_2.csv')
X = df.drop(['loss', 'id'], 1)
X_test = df_test.drop(['id'], 1)
y = df.loss

dtest = xgb.DMatrix(X_test.values)

y_oob = np.zeros(X.shape[0])
y_pred = np.zeros(df_test.shape[0])
cv_score = 0
best_iter = 0


dtrain = xgb.DMatrix(X.values, target_transform(y))

num_boost_round = int(3847 / 0.9)

xgbm = xgb.train(params, dtrain, num_boost_round,
                 verbose_eval=True,
                 obj=fair_obj,
                 feval=xgb_eval)

y_pred = target_inverse_transform(xgbm.predict(dtest))

submission = df_test[['id']].copy()
submission['loss'] = y_pred
submission.to_csv('../submissions/12_12_feat_xgb_root.csv', index=False)
