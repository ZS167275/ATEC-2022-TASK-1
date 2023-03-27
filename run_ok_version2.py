# -*- coding: utf-8 -*-
import json
import os
import random
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    print('get_second_feature_count')
    for col1 in key1:
        for col2 in key2:
            t = gdata.groupby([col1, col2])['dt'].count().reset_index()
            t.columns = [col1, col2] + ['{}_{}_{}_count'.format(sunffix, col1, col2)]
            data = pd.merge(data, t, on=[col1, col2], how='left', copy=False)
    return data


# 获取任意两个特征的交叉统计
def get_second_feature_corss(data, gdata, key1, key2, func=['count', 'nunique'], sunffix='gdata'):
    print('get_second_feature_corss')
    for col1 in key1:
        for col2 in key2:
            t = gdata.groupby([col1]).agg({col2: func}).reset_index()
            t.columns = [
                x[0] if x[1] == '' else '{}_'.format(sunffix) + x[0] + '_' + col1 + '_' + x[1] + '_' + 'cross'
                for x in
                t.columns]
            data = pd.merge(data, t, on=[col1], how='left', copy=False)
    return data


# 流势word2vec迭代更新，每次随机更新一部分新加入的userid和itemid完成对输入的id的更新迭代
def auto_word2vec(data, dt, key1='user_id', key2='item_id', is_base=True, emb=10):
    data['sp_id'] = [x for x in range(0, len(data))]
    if is_base:
        t = data[data["dt"] == dt][[key1, key2]].copy()
        t[key1] = t[key1].astype(str)
        t[key2] = 'w' + t[key2].astype(str)
        t = t[[key1, key2]].values.tolist()
        sentences = t
        model = Word2Vec(min_count=1, vector_size=emb)
        model.build_vocab(sentences)  # prepare the model vocabulary
        model.train(sentences, total_examples=model.corpus_count, epochs=10)  # train word vectors
        a_emb = []
        for index, t in enumerate(sentences):
            try:
                v = model.wv.get_vector(str(t[0]))
            except:
                v = np.zeros(shape=(1, emb))
            a_emb.append(np.array(v).reshape(1, -1))
        tmp_user_w2v = np.array(a_emb).reshape(-1, emb)
        data = data[data['dt'] == dt]
        w2v = data[['sp_id']]
        w2v['sp_id'] = w2v['sp_id'].astype(int)
        # print(tmp_user_w2v)
        for i in range(0, emb):
            w2v.loc[:, '{}_{}_v2v_{}'.format(key1, key2, i)] = tmp_user_w2v[:, i]
        model.save("./output/data/{}_{}.bin".format(key1, dt))  # 保存完整的模型,除包含词-向量,还保存词频等训练所需信息
        model.wv.save_word2vec_format("./output/data/{}_{}.dict".format(key1, dt))  # 保存的模型仅包含词-向量信息
        return w2v
    else:
        print('update dt {} word2vec'.format(dt))
        t = data[data["dt"] == dt][[key1, key2]].copy()
        t[key1] = t[key1].astype(str)
        t[key2] = 'w' + t[key2].astype(str)
        t = t[[key1, key2]].values.tolist()
        data = data[data['dt'] == dt]
        w2v = data[['sp_id']]

        model = Word2Vec.load("./output/data/{}_{}.bin".format(key1, dt - 1))
        a_s = []
        a_emb = []
        shif_q = random.randint(10110, 20220)
        print(len(t), shif_q)
        for index, sentences in enumerate(t):
            if (index + 1) % shif_q != 0:
                a_s.append(sentences)
            else:
                a_s.append(sentences)
                # 逐条更新，逐条记录
                model.build_vocab(a_s, update=True)
                model.train(a_s, epochs=10, total_examples=model.corpus_count)
                for index, sentences in enumerate(a_s):
                    try:
                        v = model.wv.get_vector(str(sentences[0]))
                    except:
                        v = np.zeros(shape=(1, emb))
                    a_emb.append(np.array(v).reshape(1, -1))
                a_s = []
        if len(a_s) != 0:
            # 逐条更新，逐条记录
            model.build_vocab(a_s, update=True)
            model.train(a_s, epochs=10, total_examples=model.corpus_count)
            for index, sentences in enumerate(a_s):
                try:
                    v = model.wv.get_vector(str(sentences[0]))
                except:
                    v = np.zeros(shape=(1, emb))
                a_emb.append(np.array(v).reshape(1, -1))
        tmp_user_w2v = np.array(a_emb).reshape(-1, emb)
        model.save("./output/data/{}_{}.bin".format(key1, dt))  # 保存完整的模型,除包含词-向量,还保存词频等训练所需信息
        model.wv.save_word2vec_format("./output/data/{}_{}.dict".format(key1, dt))  # 保存的模型仅包含词-向量信息
        w2v['sp_id'] = w2v['sp_id'].astype(int)
        # print(tmp_user_w2v)
        for i in range(0, emb):
            w2v.loc[:, '{}_{}_v2v_{}'.format(key1, key2, i)] = tmp_user_w2v[:, i]
        return w2v


# 获取任意两个特征的交叉统计
def get_second_feature_corss_tmp(gdata, key1, key2, func=['count', 'nunique'], sunffix='gdata'):
    t = gdata.groupby([key1]).agg({key2: func}).reset_index()
    t.columns = [
        x[0] if x[1] == '' else '{}_'.format(sunffix) + x[0] + '_' + key1 + '_' + x[1] + '_' + 'cross'
        for x in
        t.columns]
    return t


def k_flod_get_train_result(total_to_model, test=6):
    train_to_model = total_to_model[~total_to_model['dt'].isin([test])]
    test_to_model = total_to_model[total_to_model['dt'].isin([test])]

    K_train = train_to_model[feature]
    k_train = train_to_model[['label']]

    X_test = test_to_model[feature]

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 2022,
        'force_col_wise': 'true'
    }
    callbacks = [log_evaluation(period=250), early_stopping(stopping_rounds=50)]
    feat_imp = pd.DataFrame()
    feat_imp['name'] = feature
    y_val_pred = np.zeros((k_train.shape[0], 1))
    y_test_pred = np.zeros((X_test.shape[0], 5))
    for index, (tr_idx, vl_idx) in enumerate(skf.split(K_train, k_train)):
        X_train, y_train = K_train.iloc[tr_idx], k_train.iloc[tr_idx]
        X_valid, y_valid = K_train.iloc[vl_idx], k_train.iloc[vl_idx]

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)

        print('Starting training...')
        # feature_name and categorical_feature
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1500,
                        valid_sets=[lgb_eval],
                        callbacks=callbacks
                        )
        y_val_pred[vl_idx, :] = gbm.predict(X_valid).reshape(-1, 1)
        y_test_pred[:, index] = gbm.predict(X_test)
        feat_imp[index] = gbm.feature_importance()

    u_gauc = gauc(k_train.values.reshape(1, -1)[0], y_val_pred.reshape(1, -1)[0], train_to_model['user_id'].values)
    i_gauc = gauc(k_train.values.reshape(1, -1)[0], y_val_pred.reshape(1, -1)[0], train_to_model['item_id'].values)

    ui_gauc = (u_gauc + i_gauc) / 2

    print('ui_gauc', ui_gauc, 'u_gauc', u_gauc, 'i_gauc', i_gauc)

    return np.mean(y_test_pred, axis=1), feat_imp


def get_train_result(total_to_model, val=5, test=6):
    train_to_model = total_to_model[~total_to_model['dt'].isin([val, test])]
    val_to_model = total_to_model[total_to_model['dt'].isin([val])]
    test_to_model = total_to_model[total_to_model['dt'].isin([test])]

    X_train = train_to_model[feature]
    y_train = train_to_model[['label']]

    X_val = val_to_model[feature]
    y_val = val_to_model[['label']]

    X_test = test_to_model[feature]

    # X_train = csr_matrix((X_train))
    # X_val = csr_matrix((X_val))
    # X_test = csr_matrix((X_test))

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    # base model
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 2022,
        'force_col_wise': 'true'
    }
    callbacks = [log_evaluation(period=50), early_stopping(stopping_rounds=50)]

    print('Starting training...')
    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=[lgb_eval],
                    callbacks=callbacks
                    )

    y_val_pred = gbm.predict(X_val)

    feat_imp = pd.DataFrame()

    feat_imp['name'] = feature
    feat_imp['imp'] = gbm.feature_importance()

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
    y_test_pred = gbm.predict(X_test)
    return y_test_pred, feat_imp


if __name__ == '__main__':
    input_data_path = './home/admin/workspace/job/input'  # 数据所在文件夹目录
    output_predictions_path = './output/predictions/predictions.jsonl'  # 打分文件输出地址
    os.environ.setdefault('LOCAL', '1')
    if os.environ.get('LOCAL'):
        print('get train.jsonl')
        train_data_path = os.path.join(input_data_path, 'train.jsonl')
        train_samples = []
        with open(train_data_path) as f:
            for line in f:
                sample = json.loads(line)
                train_samples.append(sample)
        train_df = pd.DataFrame.from_records(train_samples)

        print('get test.jsonl')
        test_data_path = os.path.join(input_data_path, 'test.jsonl')
        test_samples = []
        with open(test_data_path) as f:
            for line in f:
                sample = json.loads(line)
                test_samples.append(sample)
        test_df = pd.DataFrame.from_records(test_samples)

        user_feat = pd.read_csv(
            os.path.join(
                input_data_path,
                'resources/user_feat.csv')).sort_values('user_id').reset_index(
            drop=True)

        item_feat = pd.read_csv(
            os.path.join(
                input_data_path,
                'resources/item_feat.csv')).sort_values('item_id').reset_index(
            drop=True)

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
        # print('w_2_v_df')
        # w_2_v_df = []
        # for dt in [1, 2, 3, 4, 5, 6]:
        #     print(dt)
        #     h_dt_w = total_data[total_data['dt'] < dt].groupby(['user_id'])['item_id'].apply(
        #         lambda x: list(x)).reset_index()
        #     h_dt_w['item_id'] = h_dt_w['item_id'].apply(lambda x: ' '.join([str(i) for i in x]))
        #     t = h_dt_w[['item_id']].values.tolist()
        #     sentences = t
        #     model = Word2Vec(min_count=1, vector_size=10)
        #     model.build_vocab(sentences)  # prepare the model vocabulary
        #     model.train(sentences, total_examples=model.corpus_count, epochs=10)  # train word vectors
        #     a_emb = []
        #     for index, sub_sentences in enumerate(sentences):
        #         for sub_sentence in sub_sentences:
        #             v = np.zeros(shape=(1, 10))
        #             for t in sub_sentence.split(' '):
        #                 try:
        #                     v += model.wv.get_vector(str(t)) / len(sub_sentence.split(' '))
        #                 except:
        #                     v += np.zeros(shape=(1, 10)) / len(sub_sentence.split(' '))
        #         a_emb.append(np.array(v).reshape(1, -1))
        #     tmp_user_w2v = np.array(a_emb).reshape(-1, 10)
        #     w2v = h_dt_w[['user_id']]
        #     w2v['user_id'] = w2v['user_id'].astype(int)
        #     for i in range(0, 10):
        #         w2v.loc[:, '{}_{}_v2v_{}'.format('user_id', 'item_id_list', i)] = tmp_user_w2v[:, i]
        #     w2v['dt'] = dt
        #     w_2_v_df.append(w2v)
        # w_2_v_df = pd.concat(w_2_v_df)
        #
        # total_data = pd.merge(total_data,w_2_v_df,on=['user_id','dt'],how='left',copy=False)
        # 构造历史的word2vec特征
        # # 生成 w2v emebding迭代特征
        # w2v_res = []
        # res = auto_word2vec(data=total_data, dt=0, key1='user_id', key2='item_id', is_base=True, emb=10)
        # w2v_res.append(res)
        # for dt in [1, 2, 3, 4, 5, 6]:
        #     res = auto_word2vec(data=total_data, dt=dt, key1='user_id', key2='item_id', is_base=False, emb=10)
        #     w2v_res.append(res)
        #
        # w2v_res_df = pd.concat(w2v_res, axis=0, ignore_index=True, copy=False)
        # total_data['sp_id'] = [x for x in range(0, len(total_data))]
        # total_data = pd.merge(total_data, w2v_res_df, on=['sp_id'], how='left', copy=False)
        # del total_data['sp_id']

        # w2v_res = []
        # res = auto_word2vec(data=total_data, dt=0, key1='item_id', key2='user_id', is_base=True, emb=10)
        # w2v_res.append(res)
        # for dt in [1, 2, 3, 4, 5, 6]:
        #     res = auto_word2vec(data=total_data, dt=dt, key1='item_id', key2='user_id', is_base=False, emb=10)
        #     w2v_res.append(res)
        #
        # w2v_res_df = pd.concat(w2v_res, axis=0, ignore_index=True, copy=False)
        # total_data['sp_id'] = [x for x in range(0, len(total_data))]
        # total_data = pd.merge(total_data, w2v_res_df, on=['sp_id'], how='left', copy=False)
        # del total_data['sp_id']
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

        total_data = pd.merge(total_data, mkt, on=['item_id'], how='left', copy=False)
        # # 读取图特征
        # # uu_edge_table = pd.read_csv(
        # #     os.path.join(input_data_path, 'resources/uu_graph.csv'))
        # # source_entity_id,target_entity_id
        # t = mkt_edge_table[['source_entity_id', 'target_entity_id']].copy()
        # t['source_entity_id'] = t['source_entity_id'].astype(str)
        # t['target_entity_id'] =t['target_entity_id'].astype(str)
        # t = t[['source_entity_id', 'target_entity_id']].values.tolist()
        # sentences = t
        # model = Word2Vec(min_count=1, vector_size=10)
        # model.build_vocab(sentences)  # prepare the model vocabulary
        # model.train(sentences, total_examples=model.corpus_count, epochs=10)  # train word vectors
        # a_emb = []
        # a_user_id = []
        # for index, t in enumerate(total_data['item_id'].unique()):
        #     try:
        #         v = model.wv.get_vector(str(t))
        #         a_user_id.append(str(t))
        #     except:
        #         v = np.zeros(shape=(1, 10))
        #         a_user_id.append(str(t[0]))
        #     a_emb.append(np.array(v).reshape(1, -1))
        # w2v = pd.DataFrame()
        # tmp_user_w2v = np.array(a_emb).reshape(-1, 10)
        # w2v['item_id'] = a_user_id
        # w2v['item_id'] = w2v['item_id'].astype(int)
        # # print(tmp_user_w2v)
        # for i in range(0, 10):
        #     w2v.loc[:, '{}_{}_v2v_{}'.format('u', 'u', i)] = tmp_user_w2v[:, i]
        # #
        # total_data = pd.merge(total_data,w2v,on=['item_id'],how='left',copy=False)
        # mkt_edge_table = pd.read_csv(
        #     os.path.join(input_data_path, 'resources/mkt_kg_graph.csv'))
        #
        # f_df = []
        # for d in [0,1,2,3,4,5,6]:
        #     t = total_data[total_data['dt']==d][['item_id']].drop_duplicates(['item_id'])
        #     t_uu_edge_table = mkt_edge_table[mkt_edge_table['source_entity_id'].isin(t['item_id'].unique())]
        #     t_uu_edge_table.loc[t_uu_edge_table['target_entity_id'].isin(t['item_id'].unique()),'l'] = 1
        #     t_uu_edge_table = t_uu_edge_table.fillna(0)
        #     # 今天交互的用户，均值
        #     f = t_uu_edge_table.groupby(['source_entity_id']).agg({'l':['mean','sum']}).reset_index()
        #     f.columns = ['item_id' if x[1]=='' else 'uu' + x[0] + x[1] for x in f.columns]
        #     f['dt'] = d
        #     f_df.append(f)
        # f_df = pd.concat(f_df,ignore_index=True)
        # total_data = pd.merge(total_data,f_df,on=['item_id','dt'],how='left',copy=False)
        #
        # f_df = []
        # for d in [0,1,2,3,4,5,6]:
        #     t = total_data[total_data['dt']==d][['item_id']].drop_duplicates(['item_id'])
        #     t_uu_edge_table = mkt_edge_table[mkt_edge_table['target_entity_id'].isin(t['item_id'].unique())]
        #     t_uu_edge_table.loc[t_uu_edge_table['source_entity_id'].isin(t['item_id'].unique()),'r'] = 1
        #     t_uu_edge_table = t_uu_edge_table.fillna(0)
        #     # 今天交互的用户，均值
        #     f = t_uu_edge_table.groupby(['target_entity_id']).agg({'r':['mean','sum']}).reset_index()
        #     f.columns = ['item_id' if x[1]=='' else 'uu' + x[0] + x[1] for x in f.columns]
        #     f['dt'] = d
        #     f_df.append(f)
        # f_df = pd.concat(f_df,ignore_index=True)
        # total_data = pd.merge(total_data,f_df,on=['item_id','dt'],how='left',copy=False)

        # 用户侧基础特征 'user_id', 'gender', 'age', 'occupation', 'entity_cnt'
        user_feat = user_feat.fillna(-1)
        user_feat_f = user_feat[['user_id', 'gender', 'age', 'occupation']]
        total_data = pd.merge(total_data, user_feat_f, on=['user_id'], how='left', copy=False)
        user_feat_f_add = user_feat[['user_id', 'entity_cnt']]
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

        del item_feat_f, item_feat

        # 构造商品的类型特征
        print(total_data['dt'].unique())
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
        # [0,1,2,3,4,5,6]
        train_to_model_1 = total_data[total_data['dt'] == total_data['dt'].max() - 2].reset_index(drop=True)
        train_to_model_2 = total_data[total_data['dt'] == total_data['dt'].max() - 3].reset_index(drop=True)
        train_to_model_3 = total_data[total_data['dt'] == total_data['dt'].max() - 4].reset_index(drop=True)
        train_to_model_4 = total_data[total_data['dt'] == total_data['dt'].max() - 5].reset_index(drop=True)
        val_to_model = total_data[total_data['dt'] == total_data['dt'].max() - 1].reset_index(drop=True)
        test_to_model = total_data[total_data['dt'] == total_data['dt'].max()].reset_index(drop=True)

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


        def ratio_mean(x):
            # x_len = len(x)
            x_val = x.values
            # x_weigth = [x / x_len for x in range(1, x_len + 1)]
            return np.mean(x_val)


        def ratio_var(x):
            # x_len = len(x)
            x_val = x.values
            # x_weigth = [x / x_len for x in range(1, x_len + 1)]
            return np.std(x_val)


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

        val_to_model = get_second_feature_corss(val_to_model,
                                                total_data[total_data['dt'] < val_to_model['dt'].max()],
                                                key1=history_feature_number_key1,
                                                key2=history_feature_number_key2,
                                                func=['mean', 'var'],
                                                sunffix='ctr')

        test_to_model = get_second_feature_corss(test_to_model,
                                                 total_data[total_data['dt'] < test_to_model['dt'].max()],
                                                 key1=history_feature_number_key1,
                                                 key2=history_feature_number_key2,
                                                 func=['mean', 'var'],
                                                 sunffix='ctr')
        print('finish base history ctf feature')

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

        total_data_for_val = get_second_feature_corss(
            total_data[total_data['dt'] < val_to_model['dt'].max()][
                ['user_id', 'dt'] + history_feature_number_key1 + history_feature_number_key2],
            total_data[total_data['dt'] < val_to_model['dt'].max()][
                history_feature_number_key1 + history_feature_number_key2],
            key1=history_feature_number_key1,
            key2=history_feature_number_key2,
            func=['mean', 'var'],
            sunffix='tmp')

        total_data_for_val['tmp_label_item_id_mean_cross'] = total_data_for_val[
                                                                 'tmp_label_item_id_mean_cross'] / \
                                                             (total_data_for_val['dt'].max() + 1 - total_data_for_val[
                                                                 'dt'])
        total_data_for_val['tmp_label_item_id_var_cross'] = total_data_for_val['tmp_label_item_id_var_cross'] / \
                                                            (total_data_for_val['dt'].max() + 1 - total_data_for_val[
                                                                'dt'])

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
            x for x in total_data_for_train_1.columns if str(x).__contains__('tmp_')
        ]
        print(label2lable_f)
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

        val_to_model = get_second_feature_corss(val_to_model,
                                                total_data_for_val,
                                                key1=['user_id'],
                                                key2=label2lable_f,
                                                func=['mean', 'var'],
                                                sunffix='h_s')

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

        val_to_model = get_second_feature_count(val_to_model,
                                                total_data[total_data['dt'] < val_to_model['dt'].max()],
                                                key1=count_feature_key1,
                                                key2=count_feature_key2,
                                                sunffix='history')

        test_to_model = get_second_feature_count(test_to_model,
                                                 total_data[total_data['dt'] < test_to_model['dt'].max()],
                                                 key1=count_feature_key1,
                                                 key2=count_feature_key2,
                                                 sunffix='history')

        feature = [x for x in train_to_model_1.columns if x not in [
            'id', 'log_time', 'label', 'dt'
        ]]

        print(len(feature))
        print(feature)
        train_to_model = pd.concat([train_to_model_1,
                                    train_to_model_2,
                                    train_to_model_3,
                                    train_to_model_4,
                                    ], ignore_index=True, copy=False)
        train_to_model = train_to_model.fillna(-999)
        val_to_model = val_to_model.fillna(-999)
        test_to_model = test_to_model.fillna(-999)
        total_to_model = pd.concat([train_to_model, val_to_model, test_to_model])

        # y_test_pred,feature_tmp = k_flod_get_train_result(total_to_model, test=6)
        # exit()
        y_test_pred_1, feat_imp_1 = get_train_result(total_to_model, 1, 6)
        y_test_pred_2, feat_imp_2 = get_train_result(total_to_model, 2, 6)
        y_test_pred_3, feat_imp_3 = get_train_result(total_to_model, 3, 6)
        y_test_pred_4, feat_imp_4 = get_train_result(total_to_model, 4, 6)
        y_test_pred_5, feat_imp_5 = get_train_result(total_to_model, 5, 6)
        #
        y_test_pred = (y_test_pred_1 + y_test_pred_2 + y_test_pred_3 + y_test_pred_4 + y_test_pred_5) / 5

        results = []
        for idx, sample in enumerate(test_to_model['id'].values):
            results.append({
                'id': int(sample),
                'label': y_test_pred[idx],
            })

        with open(output_predictions_path, 'w') as f:
            for line in results:
                f.write(json.dumps(line, default=str) + '\n')

    if os.environ.get('LOCAL') is None:
        # 非 LOCAL 模式运行时，将 ./output/ 目录下的打分文件及 ckpt 挪到指定位置
        new_output_predictions_path = '/home/admin/workspace/job/output/predictions/predictions.jsonl'
        # new_output_model_path = '/home/admin/workspace/job/output/data/model.ckpt'

        import shutil

        # shutil.copy(output_model_path, new_output_model_path)
        shutil.copy(output_predictions_path, new_output_predictions_path)
        # print('模型 ckpt 已拷贝至：', new_output_model_path)
        print('模型打分结果已拷贝至：', new_output_predictions_path)
