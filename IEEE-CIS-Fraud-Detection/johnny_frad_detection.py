import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML


os.chdir('/Users/chenchingchun/Desktop/Data_Game/ieee-fraud-detection')
# load dataset

df_sample_submission = pd.read_csv("sample_submission.csv")

df_test_transaction = pd.read_csv("test_transaction.csv")

df_train_transaction = pd.read_csv("train_transaction.csv")

df_test_identity = pd.read_csv("test_identity.csv")

df_train_identity = pd.read_csv("train_identity.csv")

df_train_transaction.shape

df_train_transaction.groupby('isFraud').count()
# 20663/ (569877 + 20663) # 3.5% <--- imbalance data



# Merge Identity and Transaction Data

df_train = pd.merge(df_train_transaction, df_train_identity, on='TransactionID', how='left')

df_test = pd.merge(df_test_transaction, df_test_identity, on='TransactionID', how='left')

print(f'Train dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns.') # 590540 x 434
print(f'Test dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.') # 506691 x 433

# EDA

df_train['id_07'].value_counts(dropna=False, normalize=True)

plt.hist(df_train['id_01'].dropna(), bins=77);
plt.title('Distribution of id_01 variable');

plt.hist(df_train['id_07'].dropna(), bins = 77);
plt.title('Distribution of id_07 variable');

many_null_col = [col for col in df_train.columns if df_train[col].isnull().sum()/df_train.shape[0] > 0.9]
many_null_cols_test = [col for col in df_test.columns if df_test[col].isnull().sum() / df_test.shape[0] > 0.9]

list(set(many_null_col + many_null_cols_test))



# Feature Engineering
# transcation
df_train['TransactionAmt_to_mean_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])['TransactionAmt'].transform('mean')
df_train['TransactionAmt_to_mean_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])['TransactionAmt'].transform('mean')
df_train['TransactionAmt_to_std_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])['TransactionAmt'].transform('std')
df_train['TransactionAmt_to_std_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])['TransactionAmt'].transform('std')

df_test['TransactionAmt_to_mean_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])['TransactionAmt'].transform('mean')
df_test['TransactionAmt_to_mean_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])['TransactionAmt'].transform('mean')
df_test['TransactionAmt_to_std_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])['TransactionAmt'].transform('std')
df_test['TransactionAmt_to_std_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])['TransactionAmt'].transform('std')

df_train['id_02_to_mean_card1'] = df_train['id_02'] / df_train.groupby(['card1'])['id_02'].transform('mean')
df_train['id_02_to_mean_card4'] = df_train['id_02'] / df_train.groupby(['card4'])['id_02'].transform('mean')
df_train['id_02_to_std_card1'] = df_train['id_02'] / df_train.groupby(['card1'])['id_02'].transform('std')
df_train['id_02_to_std_card4'] = df_train['id_02'] / df_train.groupby(['card4'])['id_02'].transform('std')

df_test['id_02_to_mean_card1'] = df_test['id_02'] / df_test.groupby(['card1'])['id_02'].transform('mean')
df_test['id_02_to_mean_card4'] = df_test['id_02'] / df_test.groupby(['card4'])['id_02'].transform('mean')
df_test['id_02_to_std_card1'] = df_test['id_02'] / df_test.groupby(['card1'])['id_02'].transform('std')
df_test['id_02_to_std_card4'] = df_test['id_02'] / df_test.groupby(['card4'])['id_02'].transform('std')

df_train['D15_to_mean_card1'] = df_train['D15'] / df_train.groupby(['card1'])['D15'].transform('mean')
df_train['D15_to_mean_card4'] = df_train['D15'] / df_train.groupby(['card4'])['D15'].transform('mean')
df_train['D15_to_std_card1'] = df_train['D15'] / df_train.groupby(['card1'])['D15'].transform('std')
df_train['D15_to_std_card4'] = df_train['D15'] / df_train.groupby(['card4'])['D15'].transform('std')

df_test['D15_to_mean_card1'] = df_test['D15'] / df_test.groupby(['card1'])['D15'].transform('mean')
df_test['D15_to_mean_card4'] = df_test['D15'] / df_test.groupby(['card4'])['D15'].transform('mean')
df_test['D15_to_std_card1'] = df_test['D15'] / df_test.groupby(['card1'])['D15'].transform('std')
df_test['D15_to_std_card4'] = df_test['D15'] / df_test.groupby(['card4'])['D15'].transform('std')

df_train['D15_to_mean_addr1'] = df_train['D15'] / df_train.groupby(['addr1'])['D15'].transform('mean')
df_train['D15_to_mean_addr2'] = df_train['D15'] / df_train.groupby(['addr2'])['D15'].transform('mean')
df_train['D15_to_std_addr1'] = df_train['D15'] / df_train.groupby(['addr1'])['D15'].transform('std')
df_train['D15_to_std_addr2'] = df_train['D15'] / df_train.groupby(['addr2'])['D15'].transform('std')

df_test['D15_to_mean_addr1'] = df_test['D15'] / df_test.groupby(['addr1'])['D15'].transform('mean')
df_test['D15_to_mean_addr2'] = df_test['D15'] / df_test.groupby(['addr2'])['D15'].transform('mean')
df_test['D15_to_std_addr1'] = df_test['D15'] / df_test.groupby(['addr1'])['D15'].transform('std')
df_test['D15_to_std_addr2'] = df_test['D15'] / df_test.groupby(['addr2'])['D15'].transform('std')



# identity
df_train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = df_train['P_emaildomain'].str.split('.', expand=True)
df_train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = df_train['R_emaildomain'].str.split('.', expand=True)
df_test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = df_test['P_emaildomain'].str.split('.', expand=True)
df_test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = df_test['R_emaildomain'].str.split('.', expand=True)


# New add feature
## have identity or not
df_train['has_identity'] = pd.isnull(df_train['id_01'])
identity_na_percentage = pd.isnull(df_test['id_01'])

## missing value precentage
def count_of_na_value(dataset, col_name):
    dataset[col_name] = 0
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if pd.isnull(dataset.iloc[i,j]):
                dataset[col_name][i] += 1
            else:
                dataset[col_name][i] += 0
### all

count_of_na_value(df_train,'identity_na_percentage')

## date time
import datetime
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
## relative time
df_train['DT'] = df_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
df_train['DT_M'] = (df_train['DT'].dt.year - 2017) * 12 +  df_train['DT'].dt.month
df_train['DT_W'] = (df_train['DT'].dt.year - 2017) * 52 +  df_train['DT'].dt.weekofyear
df_train['DT_D'] = (df_train['DT'].dt.year - 2017) * 365 + df_train['DT'].dt.dayofyear
## exact time
df_train['DT_hour'] = df_train['DT'].dt.hour
df_train['DT_day_week'] = df_train['DT'].dt.dayofweek
df_train['DT_day'] = df_train['DT'].dt.day

## relative time
df_test['DT'] = df_test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
df_test['DT_M'] = (df_test['DT'].dt.year - 2017) * 12 +  df_test['DT'].dt.month
df_test['DT_W'] = (df_test['DT'].dt.year - 2017) * 52 +  df_test['DT'].dt.weekofyear
df_test['DT_D'] = (df_test['DT'].dt.year - 2017) * 365 + df_test['DT'].dt.dayofyear
## exact time
df_test['DT_hour'] = df_test['DT'].dt.hour
df_test['DT_day_week'] = df_test['DT'].dt.dayofweek
df_test['DT_day'] = df_test['DT'].dt.day

## R P domain
# df_train.groupby('P_emaildomain').count()



for i in range(len(df_train['has_identity'])):
    if df_train['has_identity'][i] == True:
        df_train['has_identity'][i] = 1;
    else:
        df_train['has_identity'][i] = 0



many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
many_null_cols_test = [col for col in df_test.columns if df_test[col].isnull().sum() / df_test.shape[0] > 0.9]

big_top_value_cols = [col for col in df_train.columns if df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in df_test.columns if df_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

one_value_cols = [col for col in df_train.columns if df_train[col].nunique() <= 1]
one_value_cols_test = [col for col in df_test.columns if df_test[col].nunique() <= 1]

cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))
cols_to_drop.remove('isFraud')
len(cols_to_drop)

train = df_train.drop(cols_to_drop, axis=1)
test = df_test.drop(cols_to_drop, axis=1)

cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
for col in cat_cols:
    if col in train.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
#X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
del train
test = test[["TransactionDT", 'TransactionID']]

def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)

# Cleaning infinite values to NaN
X = clean_inf_nan(X)
X_test = clean_inf_nan(X_test )

import gc
gc.collect()

from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve
# LGBM
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

# Create arrays and dataframes to store results

oof_preds = np.zeros(df_train.shape[0])

sub_preds = np.zeros(df_test.shape[0])

feature_importance_df = pd.DataFrame()

# folds = KFold(n_splits=5)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          'scale_pos_weight':3
          #'categorical_feature': cat_cols
}

if n_fold >= 0:
    dtrain = lgb.Dataset(
        train_x, label=train_y)
    dval = lgb.Dataset(
        valid_x, label=valid_y, reference=dtrain)
    bst = lgb.train(
        params, dtrain, num_boost_round=10000,
        valid_sets=[dval], early_stopping_rounds=200, verbose_eval=500)
    tmp_valid = bst.predict(valid_x, num_iteration=bst.best_iteration)
    tmp_valid.dump('input/kfold_valid_' + str(n_fold) + '.pkl')
    oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
    tmp = bst.predict(X_test, num_iteration=bst.best_iteration)
    tmp.dump('input/kfold_' + str(n_fold) + '.pkl')
    sub_preds += bst.predict(X_test, num_iteration=bst.best_iteration) / folds.n_splits

# Make the feature importance dataframe

gain = bst.feature_importance('gain')

fold_importance_df = pd.DataFrame({'feature':bst.feature_name(),

'split':bst.feature_importance('split'),

'gain':100*gain/gain.sum(),

'fold':n_fold,

}).sort_values('gain',ascending=False)

feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=666)



preds = pd.DataFrame({"TransactionID":test['TransactionID'], "isFraud":sub_preds})

# create output sub-folder
str(roc_auc_score(y, oof_preds))
preds.to_csv("output/lgb_gbdt_" + str(roc_auc_score(y, oof_preds)) + ".csv", index=False)

roc_auc_score(y, oof_preds)
