from datetime import datetime

import pandas as pd
import numpy as np

from pylightgbm.models import GBMRegressor

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

print 'Starting'

df = pd.read_csv('../run_res/feat_train_2.csv')
df_test = pd.read_csv('../run_res/feat_test_2.csv')
X = df.drop(['loss', 'id'], 1)
X_test = df_test.drop(['id'], 1)
y = df.loss


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

    model = GBMRegressor(
        num_threads=8,
        num_iterations=5000,
        verbose=False,
        early_stopping_round=25,
        bagging_seed=2016,
        metric='l1',
        learning_rate=0.05,
        max_depth=12,
        num_leaves=450,
        # num_leaves=127,
        # feature_fraction=params['feature_fraction'],
        # bagging_fraction=params['bagging_fraction'],
        feature_fraction=0.7,
        bagging_fraction=0.7,
        min_data_in_leaf=450,
        max_bin=256,
        # lambda_l1=params['lambda_l1'],
        # lambda_l2=params['lambda_l2']
    )

    model.fit(
        X_train,
        target_transform(y_train),
        test_data=[(
            X_val,
            target_transform(y_val)
        )]
    )

    y_oob[val] = target_inverse_transform(model.predict(X_val))
    y_pred += target_inverse_transform(model.predict(X_test.values))

    mae = mean_absolute_error(y_val, y_oob[val])
    cv_score += mae
    best_iter += model.best_round

    print 'MAE = {}, BEST ITER = {}'.format(
        mae,
        model.best_round
    )


df_oob = df[['id']].copy()
df_oob['loss'] = y_oob
df_oob.to_csv('../run_res/feat_lgbm_bag_oob_1.csv', index=False)

y_pred /= n_folds
submission = df_test[['id']].copy()
submission['loss'] = y_pred
submission.to_csv('../submissions/feat_lgbm_bag_1.csv', index=False)

cv_score /= n_folds
print 'Overall CV MAE = {}'.format(cv_score)

best_iter /= float(n_folds)
print 'Average best iteration = {}'.format(best_iter)
