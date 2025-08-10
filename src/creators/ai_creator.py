import os
import random
import pickle
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from src.utils.path_manager import PathManager
from src.utils.netkeiba_accesser import NetKeibaAccesser
from src.utils.notification import Notification
from src.utils.settings import Settings
from src.creators.dataset import Dataset
from src.executors.today_data import TodayData


class AiCreator:
    """ AIモデルを作成するクラス """
    TARGET_VALUE = ''
    SEED=13
    LEARN_RATE=0.01
    NUM_BOOST_ROUND=200
    STOP_COUNT = 10
    OBJECTIVE='rank_xendcg',#'lambdarank'
    METRIC = 'average_precision'
    TAEGET_LSIT = ['order', 'top1', 'top2', 'top3', 'top4', 'top5', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5']
    MUST_FEATURES = [
        '会場_ID','開催数_ID','開催日_ID','レース番号','馬番'
    ]
    TEST = None
    FEVAL = None
    THRESHOLD = 0.5
    METRIX_FUNK = None
    n_top= 1

    @staticmethod
    def create_exec_dict_place_distance(aft_df: pd.DataFrame, 
                                        is_exclusion_new_horse:bool=True, 
                                        is_exclusion_obstacle_race:bool=True) -> dict:
        """ 開催場所と距離の辞書を作成する
            Args:
                aft_df (pd.DataFrame): 学習データ
                is_exclusion_new_horse (bool): 新馬戦除外フラグ
                is_exclusion_obstacle(bool): 障害レース除外フラグ
            Returns:
                dict: 開催場所と距離の辞書
        """
        t_type = {}
        d_type = {}
        temp_df = aft_df.copy()
        #新馬戦除外判定
        if is_exclusion_new_horse==True:
            temp_df = temp_df[temp_df['クラス'] != '新馬クラス']
        print(len(temp_df.index.unique()))

        # 障害レース除外
        if is_exclusion_obstacle_race==True:
            temp_df = temp_df[temp_df['レースタイプ'] != '障害']
        print(len(temp_df.index.unique()))
        
        res_df = temp_df.groupby(['レースタイプ', '距離', '会場'], observed=True).size().reset_index(name='count')
        for index, row in res_df.iterrows():
            race_type = row['レースタイプ']
            distance = row['距離']
            place = row['会場']
            count = row['count']

            if (count == 0) or (race_type=='障害'):
                continue
            #print(f'レースタイプ: {race_type}, 距離: {int(distance)}, 会場: {place}, カウント: {count}')
            distance = Dataset.convert_distance(race_type, place, distance)
            if race_type=='ダート':
                #if distance <= 1200:
                #    distance=1200
                #elif 1800 <= distance:
                #    distance=1800

                p_type = place
                if place not in d_type.keys():
                    d_type[place] = []
                count = d_type[p_type].count(distance)
                if count > 0:
                    continue
                d_type[p_type].append(int(distance))
            elif race_type=='芝':
                #if distance <= 1200:
                #    distance=1200
                #elif 2200 <= distance:
                #    distance=2200

                p_type = place
                if place not in t_type.keys():
                    t_type[place] = []
                count = t_type[p_type].count(distance)
                if count > 0:
                    continue
                t_type[p_type].append(int(distance))

        exec_dict = {}
        exec_dict['芝'] = t_type
        exec_dict['ダート'] = d_type

        return exec_dict

    @staticmethod
    def split_dataset(df: pd.DataFrame, test_start_date: datetime, test_end_date: datetime, 
                         target: str, test_rate:float=0.2, is_exclusion_new_horse:bool=True, size:int=25000) -> tuple:
        """ 学習データの分割
            Args:
                df (pd.DataFrame): 学習データ
                test_start_date (datetime): テストデータ開始日
                test_end_date (datetime): テストデータ終了日
                target (str): 目的変数
                test_rate (float): テストデータの割合
                size (int): 学習データのサイズ
            Returns:
                tuple: 学習データ、検証データ、テストデータ
        """
        # テストデータ取得
        df['日付'] = df['日付'].dt.date
        test_df = df[(test_start_date<=df['日付'])&(df['日付']<=test_end_date)]

        # テスト開始より前のデータを取得
        train = df[df['日付']<test_start_date]
        # レースリストを取得
        race_ids = train['race_id'].unique()
        
        if len(race_ids) < size:
            length = len(race_ids)
        else:
            length = size

        print(f'length={length}')
        # 件数補正
        print(f'race_count = {len(race_ids)}')
        idx = race_ids[-length]
        print(f'{length}件目 = {idx}')
        tgt_day = train[train['race_id']==idx]['日付'].iloc[0]
        print(f'date = {tgt_day}')
        train = train[train['日付']>=tgt_day]
        race_ids = train['race_id'].unique()
        print(f'race_ids = {len(race_ids)}')
        
        # 訓練用データの最終インデックスを取得して訓練データを作成
        train_ids = race_ids[:round(len(race_ids) * (1-test_rate))]
        #train_df = df.loc[train_ids]
        train_df = df[df['race_id'].isin(train_ids)]
        # 訓練データの最終日を取得
        train_last_date = train_df['日付'].iloc[-1]
        # 訓練データと検証データに分割
        train_df = train[train['日付']<train_last_date]
        valid_df = train[train['日付']>=train_last_date]

        if is_exclusion_new_horse==False:
            # 検証データに入っている「新馬クラス」を訓練データへ移す
            make_debut = valid_df[valid_df['クラス']=='新馬クラス']
            md = len(make_debut['race_id'].unique())
            print(f'検証データ 新馬戦 = {md}')
            b = len(train_df['race_id'].unique())
            print(f'訓練データ マージ前 = {b}')
            train_df = pd.concat([train_df, make_debut])
            a=len(train_df['race_id'].unique())
            print(f'訓練データ マージ後 = {a}')
            d=len(valid_df['race_id'].unique())
            print(f'検証データ 削除前 = {d}')
            valid_df = valid_df[valid_df['クラス'] != '新馬クラス']
            v=len(valid_df['race_id'].unique())
            print(f'検証データ 削除後 = {v}')
        
        # 並び替え
        train_df = train_df.sort_values(['日付','会場_ID','開催数_ID','開催日_ID','レース番号','馬番'],ascending=[True,True,True,True,True,True])
        valid_df = valid_df.sort_values(['日付','会場_ID','開催数_ID','開催日_ID','レース番号','馬番'],ascending=[True,True,True,True,True,True])

        df_len = len(df['race_id'].unique())
        train_len = len(train_df['race_id'].unique())
        valid_len = len(valid_df['race_id'].unique())
        test_len = len(test_df['race_id'].unique())
        print(df_len, train_len, valid_len, test_len, train_len+valid_len+test_len)

        # 学習データの変換
        X_train_d = AiCreator.delete_target(train_df)
        y_train_d = train_df[target]
        # 検証データの変換
        X_valid_d = AiCreator.delete_target(valid_df)
        y_valid_d = valid_df[target]
        # テストデータの変換
        X_test_d = None
        y_test_d = None
        if test_len != 0:
            X_test_d = AiCreator.delete_target(test_df)
            y_test_d = test_df[target]

        print()
        print('train')
        print(train_df.iloc[0]['日付'], '~', train_df.iloc[-1]['日付'])
        print(f'train data = {train_len}, {train_len / (train_len+valid_len):.3f}%')
        print('valid')
        print(valid_df.iloc[0]['日付'], '~', valid_df.iloc[-1]['日付'])
        print(f'valid data = {valid_len}, {valid_len / (train_len+valid_len):.3f}%')
        if test_len != 0:
            print('test')
            print(test_df.iloc[0]['日付'], '~', test_df.iloc[-1]['日付'])
            print(f'test data = {test_len}, {test_len / (train_len+valid_len+test_len):.3f}%')
        print()

        return X_train_d, y_train_d, X_valid_d, y_valid_d, X_test_d, y_test_d

    @staticmethod
    def threshold_function(array: pd.DataFrame, threshold: float=0.5) -> pd.DataFrame:
        """ 閾値処理
            Args:
                array (pd.DataFrame): 予測値
                threshold (float): 閾値
            Returns:
                pd.DataFrame: 0 or 1のDataFrame
        """
        return (array >= threshold).astype(int)

    @staticmethod
    def cal_f1_score(y_true: pd.DataFrame, data: pd.DataFrame) -> tuple:
        """ F1スコア計算
            Args:
                y_true (pd.DataFrame): 正解ラベル
                data (pd.DataFrame): 学習データ
            Returns:
                str: 評価指標名
                float: 評価値
                bool: True
        """
        if len(AiCreator.train) == len(y_true):
            AiCreator.train['pred'] = y_true
            target_df = AiCreator.train[['pred']]
        else:
            AiCreator.valid['pred'] = y_true
            target_df = AiCreator.valid[['pred']]
            
        median = target_df.groupby('race_id', sort=False, observed=True).apply(
            lambda group: np.median(np.partition(group['pred'].values, -AiCreator.n_top)[-AiCreator.n_top:])
        )
        pred_labels = AiCreator.threshold_function(y_true, median.median())#AiCreator.THRESHOLD)
        return PathManager.METRIC, f1_score(data.get_label(), pred_labels), True

    @staticmethod
    def cal_precision(y_true: pd.DataFrame, data: pd.DataFrame) -> tuple:
        """ 適合率計算
            Args:
                y_true (pd.DataFrame): 正解ラベル
                data (pd.DataFrame): 学習データ
            Returns:
                str: 評価指標名
                float: 評価値
                bool: True
        """
        if len(AiCreator.train) == len(y_true):
            AiCreator.train['pred'] = y_true
            target_df = AiCreator.train[['pred']]
        else:
            AiCreator.valid['pred'] = y_true
            target_df = AiCreator.valid[['pred']]
        median = target_df.groupby('race_id', sort=False, observed=True).apply(
            lambda group: np.median(np.partition(group['pred'].values, -AiCreator.n_top)[-AiCreator.n_top:])
        )
        pred_labels = AiCreator.threshold_function(y_true, median.median())#AiCreator.THRESHOLD)
        return PathManager.METRIC, precision_score(data.get_label(), pred_labels, zero_division=0), True

    @staticmethod
    def objective_class(trial) -> float:
        """ Optunaの目的関数
            Args:
                trial (optuna.Trial): Optunaの試行
            Returns:
                float: 評価値
        """
        AiCreator.param = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': AiCreator.METRIC,
            'lambda_l1' : trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
            'lambda_l2' : trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 16, AiCreator.NUM_LEAVES),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 4, 128),
            'max_depth': trial.suggest_int('max_depth', 3, AiCreator.MAX_DEPTH),
            'random_state': AiCreator.SEED,
            'verbosity': -1,
            'feature_pre_filter': False,
            'is_unbalance': True,
            'num_threads': 4,
            }
        if PathManager.METRIC == 'f1_score' or PathManager.METRIC == 'precision':
            metoric = PathManager.METRIC
        else:
            metoric = AiCreator.METRIC
        pruning_callback = LightGBMPruningCallback(trial, metoric, 'Valid')
        model = lgb.train(
            params=AiCreator.param,                                # ハイパーパラメータをセット
            train_set=AiCreator.lgb_train,                         # 訓練データを訓練用にセット
            valid_sets=[AiCreator.lgb_train, AiCreator.lgb_valid], # 訓練データとテストデータをセット
            valid_names=['Train', 'Valid'],                        # データセットの名前をそれぞれ設定
            num_boost_round=AiCreator.NUM_BOOST_ROUND,             # 計算回数
            callbacks = [
                lgb.early_stopping(stopping_rounds=AiCreator.STOP_COUNT, verbose=False),
                pruning_callback
            ], # アーリーストッピング設定
            feval=AiCreator.FEVAL
        )
        
        """if PathManager.METRIC == 'precision':
            preds = model.predict(AiCreator.x_valid, num_iteration=model.best_iteration)
            AiCreator.valid['pred'] = preds
            target_df = AiCreator.valid[['pred']]
            median = target_df.groupby(level=0).apply(
                lambda group: np.median(np.partition(group['pred'].values, -AiCreator.n_top)[-AiCreator.n_top:])
            )
            data = median.median()
            pred_labels = AiCreator.threshold_function(preds, data)
            return AiCreator.METRIX_FUNK(AiCreator.y_valid, pred_labels)"""
        
        if PathManager.METRIC == 'f1_score' or PathManager.METRIC == 'precision':
            return model.best_score['Valid'][PathManager.METRIC]
        else:
            return model.best_score['Valid'][AiCreator.METRIC]

        '''
        # f1_score 計算
        #f1 = f1_score(AiCreator.y_valid, pred_labels)
        #return f1

        # acu_score 計算
        #y_prob = model.predict(AiCreator.x_valid)
        #y_pred = np.round(y_prob)
        #return roc_auc_score(
        #    np.round(AiCreator.y_valid.values),
        #    np.round(y_pred)
        #)
        
        # PR-AUC を計算
        pr_auc = average_precision_score(AiCreator.y_valid, preds)
        # PR-AUC を返す
        return pr_auc

        preds = model.predict(AiCreator.x_valid, num_iteration=model.best_iteration)
        median = AiCreator.get_median(preds)
        pred_labels = AiCreator.threshold_function(preds, median.median())
        # 適合率計算
        precision = precision_score(AiCreator.y_valid, pred_labels)
        return precision'''

    @staticmethod
    def create_class_model(df: pl.DataFrame, start: datetime, end: datetime, target: str, 
                           race_type: str, place: str, distance: int, rate: float=0.2, 
                           is_exclusion_new_horse:bool=True, is_pickle:bool=True) -> None:
        """ モデル作成
            Args:
                df (pd.DataFrame): 学習データ
                start (datetime): 学習開始日
                end (datetime): 学習終了日
                target (str): 目標変数
                race_type (str): レースタイプ
                place (str): 会場
                distance (int): 距離
                rate (float): 学習データの割合
                is_exclusion_new_horse (bool): 新馬除外フラグ
                is_pickle (bool): pickleフラグ
            Returns:
                None
        """
        AiCreator.TARGET_VALUE = target
        AiCreator.OBJECTIVE='binary'
        AiCreator.NUM_BOOST_ROUND = 500
        AiCreator.STOP_COUNT = 50
        AiCreator.TRIALS = 200
        AiCreator.WARMUP_STEPS = 15
        AiCreator.STARTUP_TRIALS = 10
        AiCreator.x_train = None
        AiCreator.y_train = None
        AiCreator.x_valid = None
        AiCreator.y_valid = None
        AiCreator.lgb_train = None
        AiCreator.lgb_valid = None

        if PathManager.METRIC == 'f1_score' or PathManager.METRIC == 'precision':
            AiCreator.METRIC = 'None'
        elif PathManager.METRIC == 'PR_AUC':
            AiCreator.METRIC = 'average_precision'
        else:
            AiCreator.METRIC = 'binary_logloss'
            
        feval=None
        if PathManager.METRIC == 'f1_score':
            AiCreator.FEVAL = AiCreator.cal_f1_score
            AiCreator.METRIX_FUNK = f1_score
            mapping = {
                'top1': 0.65,
                'top2': 0.65,
                'top3': 0.75,
                'top4': 0.7,
                'top5': 0.7
            }
            AiCreator.THRESHOLD = mapping[AiCreator.TARGET_VALUE]
        elif PathManager.METRIC == 'precision':
            AiCreator.FEVAL = AiCreator.cal_precision
            AiCreator.METRIX_FUNK = precision_score
            mapping = {
                'top1': 0.4,
                'top2': 0.5,
                'top3': 0.6,
                'top4': 0.7,
                'top5': 0.7
            }
            AiCreator.THRESHOLD = mapping[AiCreator.TARGET_VALUE]
        else:
            AiCreator.FEVAL = None
            AiCreator.METRIX_FUNK = None
            AiCreator.THRESHOLD = 0.5
            
        top_n_mapping = {
            'top1': 2,
            'top2': 3,
            'top3': 5,
            'top4': 7,
            'top5': 9
        }
        AiCreator.n_top = top_n_mapping[AiCreator.TARGET_VALUE]
        
        Settings.set_seed(13)
        if is_pickle:
            target_df = Dataset.extract_target_dataset_pd(df, race_type, place, distance, is_exclusion_new_horse)
            target_df = Dataset.convert_column_type_pd(target_df, True)
        else:
            target_df = Dataset.extract_target_dataset_pl(df, race_type, place, distance, is_exclusion_new_horse)
            target_df = Dataset.convert_column_type_pl(target_df, True, True)

        X_train_d, y_train_d, X_valid_d, y_valid_d, X_test_d, y_test_d = Dataset.split_dataset(target_df, start, end, target, rate, is_exclusion_new_horse)
        
        AiCreator.x_train = X_train_d
        AiCreator.y_train = y_train_d
        AiCreator.x_valid = X_valid_d
        AiCreator.y_valid = y_valid_d

        AiCreator.train = pd.DataFrame(index=X_train_d['race_id'])
        AiCreator.valid = pd.DataFrame(index=X_valid_d['race_id'])

        # クエリ作成
        lst_train = AiCreator.x_train.groupby('race_id', sort=False, observed=True).size().to_list()
        lst_valid = AiCreator.x_valid.groupby('race_id', sort=False, observed=True).size().to_list()
    
        AiCreator.lgb_train = lgb.Dataset(AiCreator.x_train, label=AiCreator.y_train, group=lst_train, weight=compute_sample_weight(class_weight='balanced', y=AiCreator.y_train.values).astype('float32'))
        AiCreator.lgb_valid = lgb.Dataset(AiCreator.x_valid, label=AiCreator.y_valid, group=lst_valid, reference=AiCreator.lgb_train, weight=np.ones(len(AiCreator.x_valid)).astype('float32'))

        AiCreator.NUM_LEAVES = 512
        AiCreator.MAX_DEPTH = 32

        # optuna 実行
        pruner = optuna.pruners.MedianPruner(n_startup_trials=AiCreator.STARTUP_TRIALS, n_warmup_steps=AiCreator.WARMUP_STEPS)
        sampler = optuna.samplers.TPESampler(n_startup_trials=AiCreator.STARTUP_TRIALS, seed=AiCreator.SEED)
        print('create_study')
        study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
        print('optimize')
        study.optimize(AiCreator.objective_class, n_trials=AiCreator.TRIALS)

        print("=======ベストパラメータ========")
        print(study.best_params)
        print()
        
        # 学習
        print("学習")
        param = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': AiCreator.METRIC,
            'verbosity': -1,
            'random_state': AiCreator.SEED,
            'deterministic': True,
            'is_unbalance': True,
            }
        param.update(study.best_params)
        print(param)
        
        # 学習用に重みを調整
        AiCreator.lgb_train = None
        AiCreator.lgb_valid = None
        AiCreator.lgb_train = lgb.Dataset(AiCreator.x_train, label=AiCreator.y_train, group=lst_train, weight=compute_sample_weight(class_weight='balanced', y=AiCreator.y_train.values).astype('float32'))
        AiCreator.lgb_valid = lgb.Dataset(AiCreator.x_valid, label=AiCreator.y_valid, group=lst_valid, reference=AiCreator.lgb_train, weight=compute_sample_weight(class_weight='balanced', y=AiCreator.y_valid.values).astype('float32'))
        evals={}
        model = lgb.train(
                            params=param,                                          # ハイパーパラメータをセット
                            train_set=AiCreator.lgb_train,                         # 訓練データを訓練用にセット
                            valid_sets=[AiCreator.lgb_train, AiCreator.lgb_valid], # 訓練データとテストデータをセット
                            valid_names=['Train', 'Valid'],                        # データセットの名前をそれぞれ設定
                            num_boost_round=AiCreator.NUM_BOOST_ROUND,             # 計算回数
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=AiCreator.STOP_COUNT), # アーリーストッピング設定
                                lgb.record_evaluation(evals)
                            ],
                            feval=AiCreator.FEVAL
                        )

        # パラメータ保存
        param_path = PathManager.get_param_path(start, end, AiCreator.OBJECTIVE, target, race_type, place, distance)
        file_path = Path(param_path)
        if os.path.isdir(file_path.parent) == False:
            print(f'create is {file_path.parent} folder')
            os.makedirs(file_path.parent)
        pickle.dump(param, open(param_path, 'wb'))

        # 評価保存
        eval_path = PathManager.get_eval_path(start, end, AiCreator.OBJECTIVE, target, race_type, place, distance)
        file_path = Path(eval_path)
        if os.path.isdir(file_path.parent) == False:
            print(f'create is {file_path.parent} folder')
            os.makedirs(file_path.parent)
        pickle.dump(evals, open(eval_path, 'wb'))

        # モデル保存
        model_path = PathManager.get_model_path(start, end, AiCreator.OBJECTIVE, target, race_type, place, distance)
        file_path = Path(model_path)
        if os.path.isdir(file_path.parent) == False:
            print(f'create is {file_path.parent} folder')
            os.makedirs(file_path.parent)
        pickle.dump(model, open(model_path, 'wb'))

        print()

    @staticmethod
    def execute_class_optuna(start: int, end: int, is_pickle: bool, create_days: list, is_exclusion_new_horse:bool=True) -> None:
        """ AIモデルの作成を実行するメソッド
            Args:
                start (int): 開始日
                end (int): 終了日
                is_pickle (bool): pickleを使用するかどうか
                create_days (list): 作成する日付のリスト
                is_exclusion_new_horse (bool): 新馬除外フラグ
            Returns:
                None
        """
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rate=0.25
        for target_value in tqdm(['top3', 'top2', 'top1']):
            for date in create_days:
                learn_df = Dataset.read_dataset(start, end, is_pickle)

                NetKeibaAccesser.run(TodayData.scrape_create_ai_list_split, days=date)
                for race_type, places in TodayData.exec_dict.items():
                    for place, distances in places.items():
                        if len(distances) == 0:
                            continue
                        for distance in distances:
                            print(date[0], date[-1], race_type, place, distance)
                            #AiCreator.create_class_model_focal_loss(df, date[0], date[-1], target_value, race_type, place, distance, rate, is_exclusion_new_horse)
                            AiCreator.create_class_model(learn_df, date[0], date[-1], target_value, race_type, place, distance, rate, is_exclusion_new_horse, is_pickle)

    @staticmethod
    def top3_exact_match_eval(preds, train_data):
        y_true = train_data.get_label()
        group = train_data.get_group()

        # groupごとに分割
        preds_split = np.split(preds, np.cumsum(group)[:-1])
        y_true_split = np.split(y_true, np.cumsum(group)[:-1])

        correct = 0
        total = len(group)

        for y_t, y_p in zip(y_true_split, preds_split):
            top3_true = list(np.argsort(-y_t)[:3])  # 正解の上位3インデックス
            top3_pred = list(np.argsort(-y_p)[:3])  # 予測の上位3インデックス
            if top3_true == top3_pred:
                correct += 1

        score = correct / total
        return 'top3_exact_match', score, True  # Trueは「大きいほど良い」指標

    @staticmethod
    def objective_rank(trial):
        param = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': AiCreator.OBJECTIVE,
            'metric': 'ndcg',
            'eval_at': [3, 2, 1],
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 4, 128),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
            'random_state': AiCreator.SEED,
            'verbosity': -1,
            'num_threads': 4,
            'feature_pre_filter': False,
        }
        pruning_callback = LightGBMPruningCallback(trial, 'ndcg@3', 'Valid')
        model = lgb.train(
            params=param,
            train_set=AiCreator.lgb_train,
            valid_sets=[AiCreator.lgb_train, AiCreator.lgb_valid],
            valid_names=['Train', 'Valid'],
            #feval=AiCreator.top3_exact_match_eval,
            num_boost_round=AiCreator.NUM_BOOST_ROUND,
            callbacks=[
                lgb.early_stopping(stopping_rounds=AiCreator.STOP_COUNT, verbose=False, first_metric_only=True),
                pruning_callback
            ]
        )
        return model.best_score['Valid']['ndcg@3']

    @staticmethod
    def create_rank_model(df: pd.DataFrame, start: datetime, end: datetime, objective: str, target: str,
                          race_type: str, place: str, distance: int, rate: float=0.2,
                          is_exclusion_new_horse: bool = True, is_pickle: bool = True) -> None:
        """ ランキング学習モデルをOptunaでハイパーパラメータ調整し作成する
            Args:
                df (pd.DataFrame): 学習データ
                start (datetime): 学習開始日
                end (datetime): 学習終了日
                objective (str): 目的関数
                target (str): 目標変数
                race_type (str): レースタイプ
                place (str): 会場
                distance (int): 距離
                rate (float): 学習データの割合
                is_exclusion_new_horse (bool): 新馬除外フラグ
                is_pickle (bool): pickleを使用するかどうか
            Returns:
                None
        """
        AiCreator.TARGET_VALUE = target
        AiCreator.OBJECTIVE = objective
        AiCreator.NUM_BOOST_ROUND = 500
        AiCreator.STOP_COUNT = 50
        AiCreator.TRIALS = 200
        AiCreator.WARMUP_STEPS = 15
        AiCreator.STARTUP_TRIALS = 10
        AiCreator.x_train = None
        AiCreator.y_train = None
        AiCreator.x_valid = None
        AiCreator.y_valid = None
        AiCreator.lgb_train = None
        AiCreator.lgb_valid = None

        # データ取得
        if is_pickle:
            target_df = Dataset.extract_target_dataset_pd(df, race_type, place, distance, is_exclusion_new_horse)
            target_df = Dataset.convert_column_type_pd(target_df, True)
        else:
            target_df = Dataset.extract_target_dataset_pl(df, race_type, place, distance, is_exclusion_new_horse)
            target_df = Dataset.convert_column_type_pl(target_df, True, True)

        X_train_d, y_train_d, X_valid_d, y_valid_d, _, _ = Dataset.split_dataset(target_df, start, end, target, rate, is_exclusion_new_horse)
        AiCreator.x_train = X_train_d.drop(columns=['race_id'])
        AiCreator.y_train = y_train_d
        AiCreator.x_valid = X_valid_d.drop(columns=['race_id'])
        AiCreator.y_valid = y_valid_d

        # group情報（各レースごとの出走頭数）
        group_train = X_train_d.groupby('race_id', sort=False, observed=True).size().to_list()
        group_valid = X_valid_d.groupby('race_id', sort=False, observed=True).size().to_list()

        # LightGBM Dataset
        AiCreator.lgb_train = lgb.Dataset(AiCreator.x_train, label=AiCreator.y_train, group=group_train)
        AiCreator.lgb_valid = lgb.Dataset(AiCreator.x_valid, label=AiCreator.y_valid, group=group_valid, reference=AiCreator.lgb_train)

        # Optunaでパラメータ探索
        pruner = optuna.pruners.MedianPruner(n_startup_trials=AiCreator.STARTUP_TRIALS, n_warmup_steps=AiCreator.WARMUP_STEPS)
        sampler = optuna.samplers.TPESampler(n_startup_trials=AiCreator.STARTUP_TRIALS, seed=AiCreator.SEED)
        study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
        study.optimize(AiCreator.objective_rank, n_trials=AiCreator.TRIALS)

        print("=======ベストパラメータ========")
        print(study.best_params)
        print()

        # ベストパラメータで再学習
        best_param = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': AiCreator.OBJECTIVE,
            'metric': 'ndcg',
            'eval_at': [3, 2, 1],
            'random_state': AiCreator.SEED,
            'deterministic': True,
        }
        best_param.update(study.best_params)

        AiCreator.lgb_train = None
        AiCreator.lgb_valid = None
        # group情報（各レースごとの出走頭数）
        group_train = X_train_d.groupby('race_id', sort=False, observed=True).size().to_list()
        group_valid = X_valid_d.groupby('race_id', sort=False, observed=True).size().to_list()
        AiCreator.lgb_train = lgb.Dataset(AiCreator.x_train, label=AiCreator.y_train, group=group_train)
        AiCreator.lgb_valid = lgb.Dataset(AiCreator.x_valid, label=AiCreator.y_valid, group=group_valid, reference=AiCreator.lgb_train)

        evals = {}
        model = lgb.train(
            params=best_param,
            train_set=AiCreator.lgb_train,
            valid_sets=[AiCreator.lgb_train, AiCreator.lgb_valid],
            valid_names=['Train', 'Valid'],
            num_boost_round=AiCreator.NUM_BOOST_ROUND,
            callbacks=[lgb.early_stopping(stopping_rounds=AiCreator.STOP_COUNT, verbose=False, first_metric_only=True),
                       lgb.record_evaluation(evals)]
        )

        # パラメータ保存
        param_path = PathManager.get_param_path(start, end, AiCreator.OBJECTIVE, target, race_type, place, distance)
        file_path = Path(param_path)
        if os.path.isdir(file_path.parent) == False:
            print(f'create is {file_path.parent} folder')
            os.makedirs(file_path.parent)
        pickle.dump(best_param, open(param_path, 'wb'))

        # 評価保存
        eval_path = PathManager.get_eval_path(start, end, AiCreator.OBJECTIVE, target, race_type, place, distance)
        file_path = Path(eval_path)
        if os.path.isdir(file_path.parent) == False:
            print(f'create is {file_path.parent} folder')
            os.makedirs(file_path.parent)
        pickle.dump(evals, open(eval_path, 'wb'))

        # モデル保存
        model_path = PathManager.get_model_path(start, end, AiCreator.OBJECTIVE, target, race_type, place, distance)
        file_path = Path(model_path)
        if os.path.isdir(file_path.parent) == False:
            print(f'create is {file_path.parent} folder')
            os.makedirs(file_path.parent)
        pickle.dump(model, open(model_path, 'wb'))

        print()

    @staticmethod
    def execute_rank_optuna(start: int, end: int, objective: str, is_pickle: bool, create_days: list, is_exclusion_new_horse:bool=True) -> None:
        """ AIモデルの作成を実行するメソッド
            Args:
                start (int): 開始日
                end (int): 終了日
                objective (str): 目的関数
                is_pickle (bool): pickleを使用するかどうか
                create_days (list): 作成する日付のリスト
                is_exclusion_new_horse (bool): 新馬除外フラグ
            Returns:
                None
        """
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rate=0.2
        for target_value in tqdm(['rank3']):
            for date in create_days:
                learn_df = Dataset.read_dataset(start, end, is_pickle)

                NetKeibaAccesser.run(TodayData.scrape_create_ai_list_split, days=date)
                for race_type, places in TodayData.exec_dict.items():
                    for place, distances in places.items():
                        if len(distances) == 0:
                            continue
                        for distance in distances:
                            print(date[0], date[-1], race_type, place, distance)
                            AiCreator.create_rank_model(learn_df, date[0], date[-1], objective, target_value, race_type, place, distance, rate, is_exclusion_new_horse, is_pickle)

    @staticmethod
    def create_rank_model(days: list, start_year: int, end_year: int, objective: str) -> None:
        """ モデル作成関数
            Args:
                days (list): 日付のリスト
                start_year (int): 開始年
                end_year (int): 終了年
                objective (str): 目的関数名
            Returns:
                None
        """
        Notification.send('モデル作成 開始')
        try:
            print('モデル作成')
            AiCreator.execute_rank_optuna(start_year, end_year, objective, True, days, True)
        except Exception as e:
            print(f'error: {e}')
            Notification.send(f'error: {e}')
        else:
            Notification.send('モデル作成 完了')
