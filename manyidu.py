import os
import joblib
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb

# m:模型;x:数据(特征);p：概率阈值
def model_clf_prob(m, x, p):
    return m.predict_proba(x)[:, 1] > p


def clip_normal(y):
    y = (y + 0.5).astype(int)
    y[y > 10] = 10
    y[y < 1] = 1
    return y


def predict_score(x, model_pre, model_down, model):
    y_pred_down = model_down.predict(x)
    y_pred_down = clip_normal(y_pred_down)
    y_pred = model.predict(x)
    y_pred = clip_normal(y_pred)
    y_pred[y_pred > 7] = 10
    y_pred[y_pred <= 6] = y_pred_down[y_pred <= 6]  # 降采样模型认为是0~9，则结论就是0~9

    # 低分值分类模型对结果的影响
    y_pre_pred = model_clf_prob(model_pre, x, prob_threshold)
    y_pred[y_pre_pred] = 1

    return y_pred


def read_data(data_path, cache_path):
    if os.path.exists(cache_path):
        print('读取缓存数据...')
        t_start = time()
        data = pd.read_pickle(cache_path)
        t_end = time()
        print('耗时{:.3f}秒'.format(t_end - t_start))
    else:
        print('读取原始数据...')
        t_start = time()
        data = pd.read_csv(data_path, encoding='gbk')
        data.to_pickle(cache_path)
        t_end = time()
        print('耗时{:.3f}秒'.format(t_end - t_start))
    return data


if __name__ == '__main__':
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_columns', 300)

    input_path = '../[竞赛订阅]-复赛_用户满意度多分类预测'
    model_name_lowvalue = 'consumer_satisfaction_low_value.xgb'
    model_name_downsample = 'consumer_satisfaction_down.xgb'
    model_name = 'consumer_satisfaction.xgb'
    prob_threshold = 0.6
    nan_rate = 0.5
    feature_figure_path = os.path.join(input_path, 'Figure')  # 列特征保存路径，为空则不保存

    train_data_path = os.path.join(input_path, 'train_data.csv')
    train_data_cache_path = os.path.join(input_path, 'train_data.cache')
    test_data_path = os.path.join(input_path, 'test_data.csv')
    test_data_cache_path = os.path.join(input_path, 'test_data.cache')
    data_train = read_data(train_data_path, train_data_cache_path)
    data_test = read_data(test_data_path, test_data_cache_path)
    print(data_test)
    print(data_train)
    data_test['score'] = -1
    data = pd.concat((data_train, data_test), axis=0, ignore_index=True)

    # english_chinese = pd.read_csv(os.path.join(input_path, 'dict.csv'), encoding='gbk')
    # print(data.describe())
    # print('数据类型：\n', np.unique(data.dtypes))

    # Drop Nan
    nan_cols_name = []
    for col in data.columns:
        if np.mean(data[col].isna()) > nan_rate:
            nan_cols_name.append(col)
    print('删除空值比例超过{}的列：'.format(nan_rate), nan_cols_name)
    data.drop(labels=nan_cols_name, axis=1, inplace=True)


    # 根据数据类型填充空值，顺道把只有一种值的列挑选出来
    only_one = []
    for col in data.columns:
        if data[col].dtypes == np.int64 or data[col].dtypes == np.float64:
            data[col].fillna(-1, inplace=True)
        else:
            data[col].fillna('unknown', inplace=True)
        if np.unique(data[col]).size == 1:  # 顺道记录值全部相等的列
            only_one.append(col)
    # 删掉无效特征
    data.drop(labels=['msisdn', 'support_band', 'svc_id', 'model_name', 'ue_tac_id'], inplace=True, axis=1)
    print('无效特征：', only_one)
    data.drop(labels=only_one, inplace=True, axis=1)

    # 保留前K大的数据
    col = 'model_id'
    K = 10
    col_stat = pd.value_counts(data[col], sort=True, ascending=False)
    print('col_stat =\n', col_stat)
    sel = None
    for i in range(K):
        if sel is None:
            sel = data[col] == col_stat.index[i]
        else:
            sel |= data[col] == col_stat.index[i]
    data.loc[~sel, col] = 'unknown'

    # One-hot
    cols = ['terminal_5g_type', 'model_id']
    for col in cols:
        t = pd.get_dummies(data[col], prefix=col)
        data = pd.merge(data, t, left_index=True, right_index=True)
        data.drop(labels=col, axis=1, inplace=True)

    # String 2 Number
    le = LabelEncoder()
    for col in data.columns:
        if col == 'score':
            continue
        if data[col].dtypes == object:
            data[col] = le.fit_transform(data[col])

    # 1.“是否低分”的分类模型
    data['score_high_low'] = (data['score'] <= 4).astype(int)
    sel = (data['score'] == 1) | (data['score'] == 2) | (data['score'] == 3) | (data['score'] == 8) | (data['score'] == 9)
    data_pre = data[sel]
    x = data_pre.drop(labels=['score', 'score_high_low'], inplace=False, axis=1)
    y = data_pre['score_high_low']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2022)
    model_low_path = os.path.join(input_path, model_name_lowvalue)
    if os.path.exists(model_low_path):
        print('载入低分值分类模型...')
        model_pre = joblib.load(model_low_path)
    else:
        ext = os.path.splitext(model_name_lowvalue)[-1]
        print('训练XGBoost预分类模型：', model_name_lowvalue)     # , eval_metric=['logloss']
        model_pre = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=7, min_child_weight=1, gamma=0., subsample=0.8, scale_pos_weight=1, use_label_encoder=False)

        t_start = time()
        model_pre.fit(x_train, y_train)
        t_end = time()
        print('\t预分类模型训练耗时{:.3f}秒。'.format(t_end - t_start))
        joblib.dump(model_pre, model_low_path)

        # y_train_pred = model_pre.predict(x_train)
        y_train_pred = model_clf_prob(model_pre, x_train, prob_threshold)
        # y_test_pred = model_pre.predict(x_test)
        y_test_pred = model_clf_prob(model_pre, x_test, prob_threshold)
        print('低分值分类模型-训练集混淆矩阵：\n', confusion_matrix(y_train, y_train_pred))
        print('低分值分类分类模型-训练集正确率：', accuracy_score(y_train, y_train_pred))
        print('低分值分类分类模型-测试集混淆矩阵：\n', confusion_matrix(y_test, y_test_pred))
        print('低分值分类分类模型-测试集正确率：', accuracy_score(y_test, y_test_pred))

    # 2.1 预测分数模型 - 10降采样
    sel = data['score'] > 0
    data_train = data[sel]
    sel = data_train['score'] == 10
    sel = sel[sel]
    sel = list(sel.index)
    np.random.seed(2022)
    np.random.shuffle(sel)
    sel10 = sel[:15000]
    sel = np.zeros(data_train.shape[0], dtype=bool)
    sel[sel10] = True
    sel[data_train['score'] != 10] = True
    data_train = data_train.loc[sel]
    y = data_train['score']
    print('降采样后样本个数 = \n', pd.value_counts(y))
    x = data_train.drop(labels=['score', 'score_high_low'], inplace=False, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    model_path_downsample = os.path.join(input_path, model_name_downsample)
    if os.path.exists(model_path_downsample):
        print('载入降采样分数模型...')
        model_down = joblib.load(model_path_downsample)
    else:
        ext = os.path.splitext(model_name_downsample)[-1]
        print('训练XGBoost降采样分数模型...')
        model_down = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, colsample_bytree=0.6, learning_rate=0.05, max_depth=7, alpha=1, reg_lambda=1)

        t_start = time()
        model_down.fit(x_train, y_train)
        t_end = time()
        print('建模耗时{:.3f}秒'.format(t_end - t_start))
        joblib.dump(model_down, model_path_downsample)

    # 2.2 预测分数模型 - 全量数据
    sel = data['score'] > 0
    data_train = data[sel]
    y = data_train['score']
    # print('全量数据类别个数 = \n', pd.value_counts(y))
    x = data_train.drop(labels=['score', 'score_high_low'], inplace=False, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    model_path = os.path.join(input_path, model_name)
    if os.path.exists(model_path):
        print('载入全量分数模型...')
        model = joblib.load(model_path)
    else:
        ext = os.path.splitext(model_name)[-1]
        print('训练XGBoost全量分数模型...')
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, colsample_bytree=0.6, learning_rate=0.05, max_depth=7, alpha=1, reg_lambda=1)
        t_start = time()
        model.fit(x_train, y_train)
        t_end = time()
        print('建模耗时{:.3f}秒'.format(t_end - t_start))
        joblib.dump(model, model_path)

    y_train_pred = predict_score(x_train, model_pre, model_down, model)
    err_train = y_train_pred - y_train
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    print('训练集MSE={:.5f}，MAE={:.5f}'.format(mse_train, mae_train))

    y_test_pred = predict_score(x_test, model_pre, model_down, model)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print('测试集MSE={:.5f}，MAE={:.5f}'.format(mse_test, mae_test))

    # 测试集结果
    x_test = data[data['score'] < 0].drop(labels=['score', 'score_high_low'], inplace=False, axis=1)
    y_test = predict_score(x_test, model_pre, model_down, model)
    result = pd.DataFrame(y_test, columns=['value'])
    # print(result)
    result.to_csv(os.path.join(input_path, 'result.csv'), index=False)
