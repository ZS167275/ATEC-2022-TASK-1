# -*- coding: utf-8 -*-
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import roc_auc_score

np.random.seed(2022)


# copy from deepctr
def gauc(label, pred, id_list):
    '''
    :param label: ground truth
    :param pred: predicted prob
    :param id_list: user index
    :return: gauc
    '''
    if (len(label) != len(id_list)):
        raise ValueError("impression id num should equal to the sample num," \
                         "impression id num is {}".format(len(id_list)))
    group_truth = defaultdict(lambda: [])
    group_score = defaultdict(lambda: [])
    for idx, truth in enumerate(label):
        uid = id_list[idx]
        group_truth[uid].append(label[idx])
        group_score[uid].append(pred[idx])
    group_flag = defaultdict(lambda: False)
    for uid in set(id_list):
        truths = group_truth[uid]
        # must 2 select
        if len(set(truths)) == 2:
            group_flag[uid] = True

    total_auc = 0
    total_impression = 0

    for uid in group_flag:
        if group_flag[uid]:
            total_auc += len(group_truth[uid]) * roc_auc_score(np.asarray(group_truth[uid]),
                                                               np.asarray(group_score[uid]))
            total_impression += len(group_truth[uid])
    group_auc = float(total_auc) / total_impression
    group_auc = round(group_auc, 6)
    return group_auc


# 获取任意两个特征的组合频次
def get_second_feature_count(data, gdata, key1, key2, sunffix='gdata'):
    #     print('get_second_feature_count')
    for col1 in key1:
        for col2 in key2:
            t = gdata.groupby([col1, col2])['dt'].count().reset_index()
            t.columns = [col1, col2] + ['{}_{}_{}_count'.format(sunffix, col1, col2)]
            data = pd.merge(data, t, on=[col1, col2], how='left', copy=False)
    return data


# 获取任意两个特征的交叉统计
def get_second_feature_corss(data, gdata, key1, key2, func=['count', 'nunique'], sunffix='gdata'):
    #     print('get_second_feature_corss')
    for col1 in key1:
        for col2 in key2:
            t = gdata.groupby([col1]).agg({col2: func}).reset_index()
            t.columns = [
                x[0] if x[1] == '' else '{}_'.format(sunffix) + x[0] + '_' + col1 + '_' + x[1] + '_' + 'cross'
                for x in
                t.columns]
            data = pd.merge(data, t, on=[col1], how='left', copy=False)
    return data


# 获取任意两个特征的交叉统计
def get_second_feature_corss_tmp(gdata, key1, key2, func=['count', 'nunique'], sunffix='gdata'):
    t = gdata.groupby([key1]).agg({key2: func}).reset_index()
    t.columns = [
        x[0] if x[1] == '' else '{}_'.format(sunffix) + x[0] + '_' + key1 + '_' + x[1] + '_' + 'cross'
        for x in
        t.columns]
    return t


def get_train_result(total_to_model, feature, output_model_path, val=5, sb=''):
    #     print(output_model_path)
    train_to_model = total_to_model[~total_to_model['dt'].isin([val])]
    val_to_model = total_to_model[total_to_model['dt'].isin([val])]

    X_train = train_to_model[feature]
    y_train = train_to_model[['label']]

    X_val = val_to_model[feature]
    y_val = val_to_model[['label']]

    # base model
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val)

    # specify your configurations as a dict
    params = {
        #     'bagging_freq': 5,
        #     'bagging_fraction': 0.335,
        #     'boost_from_average':'false',
        #     'boost': 'gbdt',
        #     'feature_fraction': 0.041,
        #     'learning_rate': 0.0083,
        #     'max_depth': -1,
        #     'metric':'auc',
        #     'min_data_in_leaf': 80,
        #     'min_sum_hessian_in_leaf': 10.0,
        #     'num_leaves': 13,
        #     'num_threads': 8,
        #     'tree_learner': 'serial',
        #     'objective': 'binary',
        #     'verbosity': -1,
        #     'force_col_wise': 'true'
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 64,
        'max_depth': -1,
        'learning_rate': 0.05,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 2022,
        'force_col_wise': 'true'
    }
    callbacks = [log_evaluation(period=50), early_stopping(stopping_rounds=50)]

    #     print('Starting training...')
    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=[lgb_eval],
                    callbacks=callbacks
                    )

    y_val_pred = gbm.predict(X_val)

    #     feat_imp = pd.DataFrame()

    #     feat_imp['name'] = feature
    #     feat_imp['imp'] = gbm.feature_importance()

    u_gauc = gauc(y_val.values.reshape(1, -1)[0], y_val_pred, val_to_model['user_id'].values)
    i_gauc = gauc(y_val.values.reshape(1, -1)[0], y_val_pred, val_to_model['item_id'].values)

    ui_gauc = (u_gauc + i_gauc) / 2

    print('ui_gauc', ui_gauc, 'u_gauc', u_gauc, 'i_gauc', i_gauc)

    bst_number = gbm.best_iteration + 1

    # 这里训练所有数据
    train_val = pd.concat([train_to_model, val_to_model])
    X_train_val = train_val[feature]
    y_tain_val = train_val[['label']]

    lgb_train = lgb.Dataset(X_train_val, y_tain_val)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=bst_number,
                    valid_sets=[lgb_train],
                    callbacks=callbacks
                    )
    #     print(os.path.join(output_model_path, '{}.txt'.format(val)))
    gbm.save_model(os.path.join(output_model_path, '{}{}.txt'.format(sb, val)))
    return True


class AlgSolution():

    def __init__(self):
        self.model = {}
        self.params = None

        # 定义主要的 feature 列和 label 列
        self.feat_keys = ['user_id', 'item_id', 'app', 'benefit', 'brand', 'shop', 'entity_cnt_1', 'entity_cnt_2',
                          'entity_cnt_3', 'entity_cnt_4', 'gender', 'age', 'occupation', 'category_l1_name',
                          'category_l2_name', 'category_l3_name', 'voucher_benefit_floor_amount',
                          'voucher_benefit_amount', 'age_gender', 'item_id_occupation',
                          'category_l1_name_category_l2_name', 'category_l1_name_category_l3_name',
                          'category_l2_name_category_l3_name', 'user_id_category_l1_name', 'user_id_category_l2_name',
                          'user_id_category_l3_name', 'user_id_app', 'user_id_benefit', 'user_id_brand', 'user_id_shop',
                          'occupation_category_l1_name_category_l2_name',
                          'occupation_category_l1_name_category_l3_name',
                          'occupation_category_l2_name_category_l3_name', 'ctr_label_user_id_mean_cross',
                          'ctr_label_user_id_var_cross', 'ctr_label_item_id_mean_cross', 'ctr_label_item_id_var_cross',
                          'ctr_label_gender_mean_cross', 'ctr_label_gender_var_cross', 'ctr_label_age_mean_cross',
                          'ctr_label_age_var_cross', 'ctr_label_occupation_mean_cross',
                          'ctr_label_occupation_var_cross', 'ctr_label_category_l1_name_mean_cross',
                          'ctr_label_category_l1_name_var_cross', 'ctr_label_category_l2_name_mean_cross',
                          'ctr_label_category_l2_name_var_cross', 'ctr_label_category_l3_name_mean_cross',
                          'ctr_label_category_l3_name_var_cross', 'ctr_label_user_id_category_l1_name_mean_cross',
                          'ctr_label_user_id_category_l1_name_var_cross',
                          'ctr_label_user_id_category_l2_name_mean_cross',
                          'ctr_label_user_id_category_l2_name_var_cross',
                          'ctr_label_user_id_category_l3_name_mean_cross',
                          'ctr_label_user_id_category_l3_name_var_cross', 'ctr_label_user_id_app_mean_cross',
                          'ctr_label_user_id_app_var_cross', 'ctr_label_user_id_benefit_mean_cross',
                          'ctr_label_user_id_benefit_var_cross', 'ctr_label_user_id_brand_mean_cross',
                          'ctr_label_user_id_brand_var_cross', 'ctr_label_user_id_shop_mean_cross',
                          'ctr_label_user_id_shop_var_cross', 'h_s_tmp_label_item_id_mean_cross_user_id_mean_cross',
                          'h_s_tmp_label_item_id_mean_cross_user_id_var_cross',
                          'h_s_tmp_label_item_id_var_cross_user_id_mean_cross',
                          'h_s_tmp_label_item_id_var_cross_user_id_var_cross', 'history_user_id_item_id_count',
                          'history_user_id_category_l1_name_count', 'history_user_id_category_l2_name_count',
                          'history_user_id_category_l3_name_count', 'history_gender_item_id_count',
                          'history_gender_category_l1_name_count', 'history_gender_category_l2_name_count',
                          'history_gender_category_l3_name_count', 'history_age_item_id_count',
                          'history_age_category_l1_name_count', 'history_age_category_l2_name_count',
                          'history_age_category_l3_name_count', 'history_occupation_item_id_count',
                          'history_occupation_category_l1_name_count', 'history_occupation_category_l2_name_count',
                          'history_occupation_category_l3_name_count']
        self.label_key = ['label']

    def train_model(self, input_data_path: str, output_model_path: str,
                    params: Dict, **kwargs) -> bool:
        """需要完成模型训练与特征工程，并将其结果存储于指定目录中。
            !!! 注意：
            !!!     - 此阶段不允许读取测试集相关文件
            !!!     - 此阶段不可额外读取其他预先准备的数据文件

        Args:
            input_data_path (str): 本地输入数据集目录地址
            output_model_path (str): 本地输出模型目录地址
            params (Dict): 训练输入参数。默认为 conf/default.json

        Returns:
            bool: True 成功; False 失败
        """
        self.input_data_path = input_data_path
        #         print('get train.jsonl')
        train_data_path = os.path.join(input_data_path, 'train.jsonl')
        train_samples = []
        with open(train_data_path) as f:
            for line in f:
                sample = json.loads(line)
                train_samples.append(sample)
        train_df = pd.DataFrame.from_records(train_samples)

        train_df.to_csv('{}train_df.csv'.format(output_model_path), index=False)

        user_feat = pd.read_csv(
            os.path.join(
                input_data_path,
                'resources/user_feat.csv')).sort_values('user_id').reset_index(
            drop=True)

        user_feat.to_csv('{}user_feat.csv'.format(output_model_path), index=False)

        item_feat = pd.read_csv(
            os.path.join(
                input_data_path,
                'resources/item_feat.csv')).sort_values('item_id').reset_index(
            drop=True)

        item_feat.to_csv('{}item_feat.csv'.format(output_model_path), index=False)

        train_df['dt'] = train_df['dt'].astype(int)

        # 模拟打乱顺序
        train_df = train_df.reset_index(drop=True)

        total_data = train_df.copy()

        # 打乱顺序
        tmp = []
        for dt in [x for x in range(0, total_data['dt'].max() + 1)]:
            tmp.append(total_data[total_data['dt'] == dt].sample(frac=1).reset_index(drop=True))
        total_data = pd.concat(tmp, copy=False).reset_index(drop=True)
        del tmp

        mkt_edge_table = pd.read_csv(
            os.path.join(input_data_path, 'resources/mkt_kg_graph.csv'))
        mkt_edge_table_edge = pd.concat(
            [mkt_edge_table[['source_entity_id', 'source_entity_type']].rename(columns={'source_entity_id': 'item_id',
                                                                                        'source_entity_type': 'loc'}),
             mkt_edge_table[['target_entity_id', 'target_entity_type']].rename(columns={'target_entity_id': 'item_id',
                                                                                        'target_entity_type': 'loc'}),

             ])
        mkt_edge_table_edge['flag'] = 1
        mkt = pd.pivot_table(mkt_edge_table_edge, 'flag', 'item_id', 'loc', 'sum').reset_index()
        mkt = mkt.fillna(0)
        mkt.to_csv('{}mkt.csv'.format(output_model_path), index=False)

        total_data = pd.merge(total_data, mkt, on=['item_id'], how='left', copy=False)

        # 用户侧基础特征 'user_id', 'gender', 'age', 'occupation', 'entity_cnt'
        user_feat = user_feat.fillna(-1)

        feature_dictionary_list = {}
        for index, entity_cnt in enumerate(user_feat[['user_id', 'entity_cnt']].values):
            feature_dictionary = {}
            try:
                tmp = entity_cnt[1].split(',')
                for t_tmp in tmp:
                    k, v = t_tmp.split(':')
                    t = math.floor(math.exp(float(v)))
                    feature_dictionary[k] = t
            except:
                feature_dictionary = {}
            feature_dictionary_list[entity_cnt[0]] = feature_dictionary
        res_uid = []
        res_m1 = []
        res_m2 = []
        res_m3 = []
        res_m4 = []
        res_m5 = []
        res_m6 = []
        res_m7 = []
        res_m8 = []
        res_m9 = []
        res_m10 = []
        res_m11 = []
        res_m12 = []
        res_m13 = []
        res_m14 = []
        res_m15 = []
        for fdl in feature_dictionary_list:
            tmp = feature_dictionary_list[fdl]
            m = sorted(tmp.items(), key=lambda items: items[1])
            res_uid.append(fdl)
            try:
                res_m1.append(m[-1][0])
            except:
                res_m1.append('0')
            try:
                res_m2.append(m[-2][0])
            except:
                res_m2.append('0')
            try:
                res_m3.append(m[-3][0])
            except:
                res_m3.append('0')
            try:
                res_m4.append(m[-4][0])
            except:
                res_m4.append('0')
            try:
                res_m5.append(m[-5][0])
            except:
                res_m5.append('0')
            try:
                res_m6.append(m[-6][0])
            except:
                res_m6.append('0')
            try:
                res_m7.append(m[-7][0])
            except:
                res_m7.append('0')
            try:
                res_m8.append(m[-8][0])
            except:
                res_m8.append('0')
            try:
                res_m9.append(m[-9][0])
            except:
                res_m9.append('0')
            try:
                res_m10.append(m[-10][0])
            except:
                res_m10.append('0')
            try:
                res_m11.append(m[0][0])
            except:
                res_m11.append('0')
            try:
                res_m12.append(m[1][0])
            except:
                res_m12.append('0')
            try:
                res_m13.append(m[2][0])
            except:
                res_m13.append('0')
            try:
                res_m14.append(m[3][0])
            except:
                res_m14.append('0')
            try:
                res_m15.append(m[4][0])
            except:
                res_m15.append('0')
        add_user_feature = pd.DataFrame()
        add_user_feature['user_id'] = res_uid
        add_user_feature['entity_cnt_1'] = res_m1
        add_user_feature['entity_cnt_2'] = res_m2
        add_user_feature['entity_cnt_3'] = res_m3
        add_user_feature['entity_cnt_4'] = res_m4
        add_user_feature['entity_cnt_5'] = res_m5
        add_user_feature['entity_cnt_6'] = res_m6
        add_user_feature['entity_cnt_7'] = res_m7
        add_user_feature['entity_cnt_8'] = res_m8
        add_user_feature['entity_cnt_9'] = res_m9
        add_user_feature['entity_cnt_10'] = res_m10
        add_user_feature['entity_cnt_11'] = res_m11
        add_user_feature['entity_cnt_12'] = res_m12
        add_user_feature['entity_cnt_13'] = res_m13
        add_user_feature['entity_cnt_14'] = res_m14
        add_user_feature['entity_cnt_15'] = res_m15
        add_user_feature.to_csv('{}add_user_feature.csv'.format(output_model_path), index=False)

        add_user_feature['user_id'] = add_user_feature['user_id'].astype(int)
        add_user_feature['entity_cnt_1'] = add_user_feature['entity_cnt_1'].astype(int)
        add_user_feature['entity_cnt_2'] = add_user_feature['entity_cnt_2'].astype(int)
        add_user_feature['entity_cnt_3'] = add_user_feature['entity_cnt_3'].astype(int)
        add_user_feature['entity_cnt_4'] = add_user_feature['entity_cnt_4'].astype(int)
        add_user_feature['entity_cnt_5'] = add_user_feature['entity_cnt_5'].astype(int)
        add_user_feature['entity_cnt_6'] = add_user_feature['entity_cnt_6'].astype(int)
        add_user_feature['entity_cnt_7'] = add_user_feature['entity_cnt_7'].astype(int)
        add_user_feature['entity_cnt_8'] = add_user_feature['entity_cnt_8'].astype(int)
        add_user_feature['entity_cnt_9'] = add_user_feature['entity_cnt_9'].astype(int)
        add_user_feature['entity_cnt_10'] = add_user_feature['entity_cnt_10'].astype(int)
        add_user_feature['entity_cnt_11'] = add_user_feature['entity_cnt_11'].astype(int)
        add_user_feature['entity_cnt_12'] = add_user_feature['entity_cnt_12'].astype(int)
        add_user_feature['entity_cnt_13'] = add_user_feature['entity_cnt_13'].astype(int)
        add_user_feature['entity_cnt_14'] = add_user_feature['entity_cnt_14'].astype(int)
        add_user_feature['entity_cnt_15'] = add_user_feature['entity_cnt_15'].astype(int)
        total_data = pd.merge(total_data, add_user_feature, on=['user_id'], how='left', copy=False)

        # v = sklearn.feature_extraction.DictVectorizer(sparse=True, dtype=float)
        # entity_cnt_feature = v.fit_transform([feature_dictionary_list[x] for x in feature_dictionary_list])

        user_feat_f = user_feat[['user_id', 'gender', 'age', 'occupation']]
        total_data = pd.merge(total_data, user_feat_f, on=['user_id'], how='left', copy=False)

        # 商品侧基础特征 'item_id', 'category_l1_name', 'category_l2_name', 'category_l3_name', 'voucher_benefit_floor_amount', 'voucher_benefit_amount'
        item_feat = item_feat.fillna(-1)
        item_feat_f = item_feat[[
            'item_id',
            'category_l1_name', 'category_l2_name', 'category_l3_name',
            'voucher_benefit_floor_amount', 'voucher_benefit_amount'
        ]]

        # 连续值特征的量化
        item_feat_f['voucher_benefit_floor_amount'] = item_feat_f['voucher_benefit_floor_amount'].apply(
            lambda x: np.log(2 + x))
        item_feat_f['voucher_benefit_amount'] = item_feat_f['voucher_benefit_amount'].apply(lambda x: np.log(2 + x))

        total_data = pd.merge(total_data, item_feat_f, on=['item_id'], how='left', copy=False)

        del item_feat_f, item_feat

        # 构造商品的类型特征
        #         print(total_data['dt'].unique())
        ################################# 特征拼接 #################################

        for col in [
            ['age', 'gender'],
            ['item_id', 'occupation'],
            ['category_l1_name', 'category_l2_name'],
            ['category_l1_name', 'category_l3_name'],
            ['category_l2_name', 'category_l3_name'],
            ['user_id', 'category_l1_name'],
            ['user_id', 'category_l2_name'],
            ['user_id', 'category_l3_name'],
            ['user_id', 'app'],
            ['user_id', 'benefit'],
            ['user_id', 'brand'],
            ['user_id', 'shop'],
            ['occupation', 'category_l1_name', 'category_l2_name'],
            ['occupation', 'category_l1_name', 'category_l3_name'],
            ['occupation', 'category_l2_name', 'category_l3_name'],
        ]:
            total_data['_'.join(col)] = total_data[col[0]] * 1000 + total_data[col[1]]

        train_to_model_1 = total_data[total_data['dt'] == total_data['dt'].max() - 1].reset_index(drop=True)
        train_to_model_2 = total_data[total_data['dt'] == total_data['dt'].max() - 2].reset_index(drop=True)
        train_to_model_3 = total_data[total_data['dt'] == total_data['dt'].max() - 3].reset_index(drop=True)
        train_to_model_4 = total_data[total_data['dt'] == total_data['dt'].max() - 4].reset_index(drop=True)
        train_to_model_5 = total_data[total_data['dt'] == total_data['dt'].max()].reset_index(drop=True)

        ################################# 这些特征属于有效特征 #################################
        ################################# 历史特征 #################################
        # 计算历史转化率特征，历史数据是有序列信息的
        history_feature_number_key1 = [
            'user_id',
            'item_id',
            'gender',
            'age',
            'occupation',
            'category_l1_name',
            'category_l2_name',
            'category_l3_name',
            'user_id_category_l1_name',
            'user_id_category_l2_name',
            'user_id_category_l3_name',
            'user_id_app',
            'user_id_benefit',
            'user_id_brand',
            'user_id_shop',
        ]
        history_feature_number_key2 = [
            'label'
        ]

        train_to_model_1 = get_second_feature_corss(train_to_model_1,
                                                    total_data[total_data['dt'] < train_to_model_1['dt'].max()],
                                                    key1=history_feature_number_key1,
                                                    key2=history_feature_number_key2,
                                                    func=['mean', 'var'],
                                                    sunffix='ctr')

        train_to_model_2 = get_second_feature_corss(train_to_model_2,
                                                    total_data[total_data['dt'] < train_to_model_2['dt'].max()],
                                                    key1=history_feature_number_key1,
                                                    key2=history_feature_number_key2,
                                                    func=['mean', 'var'],
                                                    sunffix='ctr')

        train_to_model_3 = get_second_feature_corss(train_to_model_3,
                                                    total_data[total_data['dt'] < train_to_model_3['dt'].max()],
                                                    key1=history_feature_number_key1,
                                                    key2=history_feature_number_key2,
                                                    func=['mean', 'var'],
                                                    sunffix='ctr')

        train_to_model_4 = get_second_feature_corss(train_to_model_4,
                                                    total_data[total_data['dt'] < train_to_model_4['dt'].max()],
                                                    key1=history_feature_number_key1,
                                                    key2=history_feature_number_key2,
                                                    func=['mean', 'var'],
                                                    sunffix='ctr')

        train_to_model_5 = get_second_feature_corss(train_to_model_5,
                                                    total_data[total_data['dt'] < train_to_model_5['dt'].max()],
                                                    key1=history_feature_number_key1,
                                                    key2=history_feature_number_key2,
                                                    func=['mean', 'var'],
                                                    sunffix='ctr')

        #         print('finish base history ctf feature')

        # 根据 item 的转化率特征 转移为 user 的转化率特征
        history_feature_number_key1 = [
            'item_id',
        ]
        history_feature_number_key2 = [
            'label',
        ]
        total_data_for_train_1 = get_second_feature_corss(
            total_data[total_data['dt'] < train_to_model_1['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < train_to_model_1['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')
        total_data_for_train_1['tmp_label_item_id_mean_cross'] = total_data_for_train_1[
                                                                     'tmp_label_item_id_mean_cross'] / \
                                                                 (total_data_for_train_1['dt'].max() + 1 -
                                                                  total_data_for_train_1['dt'])
        total_data_for_train_1['tmp_label_item_id_var_cross'] = total_data_for_train_1['tmp_label_item_id_var_cross'] / \
                                                                (total_data_for_train_1['dt'].max() + 1 -
                                                                 total_data_for_train_1['dt'])

        total_data_for_train_2 = get_second_feature_corss(
            total_data[total_data['dt'] < train_to_model_2['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < train_to_model_2['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')
        total_data_for_train_2['tmp_label_item_id_mean_cross'] = total_data_for_train_2[
                                                                     'tmp_label_item_id_mean_cross'] / \
                                                                 (total_data_for_train_2['dt'].max() + 1 -
                                                                  total_data_for_train_2['dt'])
        total_data_for_train_2['tmp_label_item_id_var_cross'] = total_data_for_train_2['tmp_label_item_id_var_cross'] / \
                                                                (total_data_for_train_2['dt'].max() + 1 -
                                                                 total_data_for_train_2['dt'])

        total_data_for_train_3 = get_second_feature_corss(
            total_data[total_data['dt'] < train_to_model_3['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < train_to_model_3['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')

        total_data_for_train_3['tmp_label_item_id_mean_cross'] = total_data_for_train_3[
                                                                     'tmp_label_item_id_mean_cross'] / \
                                                                 (total_data_for_train_3['dt'].max() + 1 -
                                                                  total_data_for_train_3['dt'])
        total_data_for_train_3['tmp_label_item_id_var_cross'] = total_data_for_train_3['tmp_label_item_id_var_cross'] / \
                                                                (total_data_for_train_3['dt'].max() + 1 -
                                                                 total_data_for_train_3['dt'])

        total_data_for_train_4 = get_second_feature_corss(
            total_data[total_data['dt'] < train_to_model_4['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < train_to_model_4['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')

        total_data_for_train_4['tmp_label_item_id_mean_cross'] = total_data_for_train_4[
                                                                     'tmp_label_item_id_mean_cross'] / \
                                                                 (total_data_for_train_4['dt'].max() + 1 -
                                                                  total_data_for_train_4['dt'])
        total_data_for_train_4['tmp_label_item_id_var_cross'] = total_data_for_train_4['tmp_label_item_id_var_cross'] / \
                                                                (total_data_for_train_4['dt'].max() + 1 -
                                                                 total_data_for_train_4['dt'])

        total_data_for_train_5 = get_second_feature_corss(
            total_data[total_data['dt'] < train_to_model_5['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < train_to_model_5['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')

        total_data_for_train_5['tmp_label_item_id_mean_cross'] = total_data_for_train_5[
                                                                     'tmp_label_item_id_mean_cross'] / \
                                                                 (total_data_for_train_5['dt'].max() + 1 -
                                                                  total_data_for_train_5[
                                                                      'dt'])
        total_data_for_train_5['tmp_label_item_id_var_cross'] = total_data_for_train_5['tmp_label_item_id_var_cross'] / \
                                                                (total_data_for_train_5['dt'].max() + 1 -
                                                                 total_data_for_train_5[
                                                                     'dt'])

        label2lable_f = [
            x for x in total_data_for_train_1.columns if str(x).__contains__('tmp_')
        ]
        #         print(label2lable_f)
        train_to_model_1 = get_second_feature_corss(train_to_model_1,
                                                    total_data_for_train_1,
                                                    key1=['user_id'],
                                                    key2=label2lable_f,
                                                    func=['mean', 'var'],
                                                    sunffix='h_s')

        train_to_model_2 = get_second_feature_corss(train_to_model_2,
                                                    total_data_for_train_2,
                                                    key1=['user_id'],
                                                    key2=label2lable_f,
                                                    func=['mean', 'var'],
                                                    sunffix='h_s')

        train_to_model_3 = get_second_feature_corss(train_to_model_3,
                                                    total_data_for_train_3,
                                                    key1=['user_id'],
                                                    key2=label2lable_f,
                                                    func=['mean', 'var'],
                                                    sunffix='h_s')

        train_to_model_4 = get_second_feature_corss(train_to_model_4,
                                                    total_data_for_train_4,
                                                    key1=['user_id'],
                                                    key2=label2lable_f,
                                                    func=['mean', 'var'],
                                                    sunffix='h_s')

        train_to_model_5 = get_second_feature_corss(train_to_model_5,
                                                    total_data_for_train_5,
                                                    key1=['user_id'],
                                                    key2=label2lable_f,
                                                    func=['mean', 'var'],
                                                    sunffix='h_s')

        # 交叉统计 历史 [a,b].count
        count_feature_key1 = ['user_id', 'gender', 'age', 'occupation']
        count_feature_key2 = ['item_id',
                              'category_l1_name',
                              'category_l2_name',
                              'category_l3_name',
                              ]
        train_to_model_1 = get_second_feature_count(train_to_model_1,
                                                    total_data[total_data['dt'] < train_to_model_1['dt'].max()],
                                                    key1=count_feature_key1,
                                                    key2=count_feature_key2,
                                                    sunffix='history')

        train_to_model_2 = get_second_feature_count(train_to_model_2,
                                                    total_data[total_data['dt'] < train_to_model_2['dt'].max()],
                                                    key1=count_feature_key1,
                                                    key2=count_feature_key2,
                                                    sunffix='history')

        train_to_model_3 = get_second_feature_count(train_to_model_3,
                                                    total_data[total_data['dt'] < train_to_model_3['dt'].max()],
                                                    key1=count_feature_key1,
                                                    key2=count_feature_key2,
                                                    sunffix='history')

        train_to_model_4 = get_second_feature_count(train_to_model_4,
                                                    total_data[total_data['dt'] < train_to_model_4['dt'].max()],
                                                    key1=count_feature_key1,
                                                    key2=count_feature_key2,
                                                    sunffix='history')

        train_to_model_5 = get_second_feature_count(train_to_model_5,
                                                    total_data[total_data['dt'] < train_to_model_5['dt'].max()],
                                                    key1=count_feature_key1,
                                                    key2=count_feature_key2,
                                                    sunffix='history')

        feature = [x for x in train_to_model_1.columns if x not in [
            'id', 'log_time', 'label', 'dt'
        ]]

        train_to_model = pd.concat([train_to_model_1,
                                    train_to_model_2,
                                    train_to_model_3,
                                    train_to_model_4,
                                    train_to_model_5,
                                    ], ignore_index=True, copy=False)
        train_to_model = train_to_model.fillna(-999)

        get_train_result(train_to_model, self.feat_keys, output_model_path, 1)
        get_train_result(train_to_model, self.feat_keys, output_model_path, 2)
        get_train_result(train_to_model, self.feat_keys, output_model_path, 3)
        get_train_result(train_to_model, self.feat_keys, output_model_path, 4)
        get_train_result(train_to_model, self.feat_keys, output_model_path, 5)

        feature = self.feat_keys
        feature.remove("user_id")
        print('item')
        get_train_result(train_to_model, feature, output_model_path, 1, 'item')
        get_train_result(train_to_model, feature, output_model_path, 2, 'item')
        get_train_result(train_to_model, feature, output_model_path, 3, 'item')
        get_train_result(train_to_model, feature, output_model_path, 4, 'item')
        get_train_result(train_to_model, feature, output_model_path, 5, 'item')

        return True

    def load_model(self, model_path: str, params: Dict, **kwargs) -> bool:
        """需要选手加载前一阶段产出的模型参数与特征文件，并初始化打分环境（例如 model.eval() 等）。
            !!! 注意：
            !!!     - 此阶段不可额外读取其他预先准备的数据文件

        Args:
            model_path (str): 本地模型路径
            params (Dict): 模型输入参数。默认为conf/default.json

        Returns:
            bool: True 成功; False 失败
        """
        index = 0
        self.params = params
        for i in [1, 2, 3, 4, 5]:
            self.model[index] = lgb.Booster(model_file=os.path.join(model_path, '{}.txt'.format(i)))
            index = index + 1
        for i in [1, 2, 3, 4, 5]:
            self.model[index] = lgb.Booster(model_file=os.path.join(model_path, '{}{}.txt'.format('item', i)))
            index = index + 1
        return True

    def predicts(self, sample_list: List[Dict], **kwargs) -> List[Dict]:
        """需要执行预测打分的流程，选手需将前一阶段加载的特征拼接在测试数据中，再送入模型执行打分。
            !!! 注意：
            !!!     - 此阶段不可额外读取其他预先准备的数据文件

        Args:
            sample_list (List[Dict]): 输入请求内容列表
            kwargs:
                __dataset_root_path (str):  测试集预测时，本地输入数据集路径
                __output_root_path (str):  本地输出路径

    Returns:
            List[Dict]: 输出预测结果列表
        """
        sample_pd = pd.DataFrame.from_records(sample_list)

        #         print('get train.jsonl')

        train_df = pd.read_csv('{}train_df.csv'.format(self.params['output_data_folder']))

        test_df = sample_pd

        mkt = pd.read_csv('{}mkt.csv'.format(self.params['output_data_folder']))
        user_feat = pd.read_csv('{}user_feat.csv'.format(self.params['output_data_folder']))
        item_feat = pd.read_csv('{}item_feat.csv'.format(self.params['output_data_folder']))
        add_user_feature = pd.read_csv('{}add_user_feature.csv'.format(self.params['output_data_folder']))

        train_df['dt'] = train_df['dt'].astype(int)
        test_df['dt'] = train_df['dt'].max() + 1

        # 模拟打乱顺序
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # 合并数据
        total_data = pd.concat([train_df, test_df], copy=False).reset_index(drop=True)
        del train_df, test_df
        # 打乱顺序
        tmp = []
        for dt in [x for x in range(0, total_data['dt'].max() + 1)]:
            tmp.append(total_data[total_data['dt'] == dt].sample(frac=1).reset_index(drop=True))
        total_data = pd.concat(tmp, copy=False).reset_index(drop=True)
        del tmp

        total_data = pd.merge(total_data, mkt, on=['item_id'], how='left', copy=False)

        # 用户侧基础特征 'user_id', 'gender', 'age', 'occupation', 'entity_cnt'
        user_feat = user_feat.fillna(-1)
        user_feat_f = user_feat[['user_id', 'gender', 'age', 'occupation']]
        total_data = pd.merge(total_data, user_feat_f, on=['user_id'], how='left', copy=False)
        del user_feat_f, user_feat

        # 商品侧基础特征 'item_id', 'category_l1_name', 'category_l2_name', 'category_l3_name', 'voucher_benefit_floor_amount', 'voucher_benefit_amount'
        item_feat = item_feat.fillna(-1)
        item_feat_f = item_feat[[
            'item_id',
            'category_l1_name', 'category_l2_name', 'category_l3_name',
            'voucher_benefit_floor_amount', 'voucher_benefit_amount'
        ]]

        # 连续值特征的量化
        item_feat_f['voucher_benefit_floor_amount'] = item_feat_f['voucher_benefit_floor_amount'].apply(
            lambda x: np.log(2 + x))
        item_feat_f['voucher_benefit_amount'] = item_feat_f['voucher_benefit_amount'].apply(lambda x: np.log(2 + x))

        total_data = pd.merge(total_data, item_feat_f, on=['item_id'], how='left', copy=False)

        add_user_feature['user_id'] = add_user_feature['user_id'].astype(int)
        total_data = pd.merge(total_data, add_user_feature, on=['user_id'], how='left', copy=False)

        del item_feat_f, item_feat

        # 构造商品的类型特征
        #         print(total_data['dt'].unique())
        ################################# 特征拼接 #################################

        for col in [
            ['age', 'gender'],
            ['item_id', 'occupation'],
            ['category_l1_name', 'category_l2_name'],
            ['category_l1_name', 'category_l3_name'],
            ['category_l2_name', 'category_l3_name'],
            ['user_id', 'category_l1_name'],
            ['user_id', 'category_l2_name'],
            ['user_id', 'category_l3_name'],
            ['user_id', 'app'],
            ['user_id', 'benefit'],
            ['user_id', 'brand'],
            ['user_id', 'shop'],
            ['occupation', 'category_l1_name', 'category_l2_name'],
            ['occupation', 'category_l1_name', 'category_l3_name'],
            ['occupation', 'category_l2_name', 'category_l3_name'],
        ]:
            total_data['_'.join(col)] = total_data[col[0]] * 1000 + total_data[col[1]]

        test_to_model = total_data[total_data['dt'] == total_data['dt'].max()].reset_index(drop=True)

        # 计算历史转化率特征，历史数据是有序列信息的
        history_feature_number_key1 = [
            'user_id',
            'item_id',
            'gender',
            'age',
            'occupation',
            'category_l1_name',
            'category_l2_name',
            'category_l3_name',
            'user_id_category_l1_name',
            'user_id_category_l2_name',
            'user_id_category_l3_name',
            'user_id_app',
            'user_id_benefit',
            'user_id_brand',
            'user_id_shop',
        ]
        history_feature_number_key2 = [
            'label'
        ]

        test_to_model = get_second_feature_corss(test_to_model,
                                                 total_data[total_data['dt'] < test_to_model['dt'].max()],
                                                 key1=history_feature_number_key1,
                                                 key2=history_feature_number_key2,
                                                 func=['mean', 'var'],
                                                 sunffix='ctr')
        #         print('finish base history ctf feature')

        # 根据 item 的转化率特征 转移为 user 的转化率特征
        history_feature_number_key1 = [
            'item_id',
        ]
        history_feature_number_key2 = [
            'label',
        ]

        total_data_for_test = get_second_feature_corss(
            total_data[total_data['dt'] < test_to_model['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < test_to_model['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')

        total_data_for_test['tmp_label_item_id_mean_cross'] = total_data_for_test[
                                                                  'tmp_label_item_id_mean_cross'] / \
                                                              (total_data_for_test['dt'].max() + 1 -
                                                               total_data_for_test['dt'])
        total_data_for_test['tmp_label_item_id_var_cross'] = total_data_for_test['tmp_label_item_id_var_cross'] / \
                                                             (total_data_for_test['dt'].max() + 1 - total_data_for_test[
                                                                 'dt'])

        label2lable_f = [
            x for x in total_data_for_test.columns if str(x).__contains__('tmp_')
        ]

        test_to_model = get_second_feature_corss(test_to_model,
                                                 total_data_for_test,
                                                 key1=['user_id'],
                                                 key2=label2lable_f,
                                                 func=['mean', 'var'],
                                                 sunffix='h_s')

        # 交叉统计 历史 [a,b].count
        count_feature_key1 = ['user_id', 'gender', 'age', 'occupation']
        count_feature_key2 = ['item_id',
                              'category_l1_name',
                              'category_l2_name',
                              'category_l3_name',
                              ]

        test_to_model = get_second_feature_count(test_to_model,
                                                 total_data[total_data['dt'] < test_to_model['dt'].max()],
                                                 key1=count_feature_key1,
                                                 key2=count_feature_key2,
                                                 sunffix='history')

        test_to_model = test_to_model.fillna(-999)
        feature2 = self.feat_keys
        feature1 = self.feat_keys
        feature2.remove("user_id")
        y_test_pred = np.zeros(shape=(1, test_to_model.shape[0]))
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            gbm = self.model[i]
            try:
                y_test_pred = y_test_pred + gbm.predict(test_to_model[feature1], num_iteration=gbm.best_iteration)
            except:
                y_test_pred = y_test_pred + gbm.predict(test_to_model[feature2], num_iteration=gbm.best_iteration)
        y_test_pred = y_test_pred / 10
        #         print(y_test_pred)
        # 使用测试数据作为 x，随机生成一个 y 仅为了使用 Trainer.test
        results = []
        for idx, sample in enumerate(test_to_model['id'].values):
            results.append({
                'id': int(sample),
                'label': y_test_pred[0][idx],
            })
        return results


if __name__ == '__main__':
    """
        以下代码仅本地测试使用，以流式打分方案提交后云端并不会执行。
    """
    config_path = './conf/default.json'
    params = json.load(open(config_path, 'r'))
    input_data_path = '/home/admin/workspace/job/input'  # 数据所在文件夹目录
    # output_predictions_path = './output/predictions/predictions.jsonl'  # 打分文件输出地址
    # input_data_path = '/home/admin/workspace/job/input'  # 数据所在文件夹目录
    output_model_path = params['output_data_folder']  # 模型存储路径

    # 执行训练逻辑
    solution = AlgSolution()
    solution.train_model(input_data_path=input_data_path,
                         output_model_path=output_model_path,
                         params=params.copy())

    # 加载 checkpoint 文件，用于打分预测
    solution = AlgSolution()
    solution.load_model(model_path=output_model_path, params=params.copy())
    #
    # # 执行测试
    test_data = []
    with open(os.path.join(input_data_path, 'test.jsonl'), 'r') as f:
        for line in f:
            sample = json.loads(line)
            test_data.append(sample)
    result = solution.predicts(sample_list=test_data, params=params.copy())
    print(result[:10])
