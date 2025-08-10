import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime
import re
import os
from typing import Union
from pathlib import Path
from tqdm import tqdm
from src.utils.path_manager import PathManager
from src.utils.netkeiba_accesser import NetKeibaAccesser
from src.utils.notification import Notification
from src.table_data.race_id import RaceId
from src.table_data.race_table import RaceTable
from src.table_data.refund_table import RefundTable
from src.table_data.pedigree_table import PedigreeTable
from src.table_data.training_table import TrainingTable
from src.organize_data.horse_rate_data import HorseRateData
from src.organize_data.past_data import PastData
from src.organize_data.statictics_data import StaticticsData
from src.organize_data.ranking_data import RankingData


class Dataset:
    """ データセットを作成するクラス """
    DELETE_NAMES = ['馬名','騎手','調教師','馬主','開催数','開催日','条件', '乗り役',
    ]
    DELETE_FEATURES = [
        '日付', '調教日', 'order', 'top1', 'top2', 'top3', 'top4', 'top5', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'
    ]
    
    @staticmethod
    def get_selected_columns(df : Union[pl.DataFrame, pd.DataFrame]) -> list:
        """ 選択するカラムを取得する
            Args:
                df (pl.DataFrame or pd.DataFrame): 元データ
            Returns:
                list: 選択するカラム
        """
        selected_cols = [
            '日付', 'race_id', 
            '枠番', '馬番', #'馬名_ID', 
            '性別', '年齢', '斤量', #'騎手_ID', 
            '脚質', '所属', 
            #'調教師_ID', '馬主_ID', 'レース名',
            'レースタイプ', 'レース周り', '距離', '馬場', '天気', '開催数_ID', '会場', '会場_ID', '開催日_ID',
            'レース番号', 'クラス', '頭数', '詳細条件_0', '詳細条件_1', '詳細条件_2', '詳細条件_3', '詳細条件_4',
            '4歳以上', '3歳以上', '3歳', '2歳', '同世代限定', '年齢条件',
            '馬齢', '指', '特指', '定量', '別定', 'ハンデ', '国際', '牡', '牝', '混', '見習騎手', '九州産馬',
            '騎手変更', 
        ]
        
        # selected_cols += ['peds_type', 'peds_00_ID', 'peds_01_ID', 'peds_32_ID']
        selected_cols += ['調教ラップ_3', '調教ラップ_2', '調教ラップ_1']
        selected_cols += ['脚色', '評価欄'] #, '評価'
        selected_cols += ['出走前馬レート', '出走前レース内馬レート']
        selected_cols += [
            'レース間隔', '総出走数', '勝利数', '連対数', '複勝数', '馬券外数', 
            '勝率', '連対率', '複勝率', '馬券外率','総出遅れ回数', '総出走回数', '総出遅れ確率'
        ]
        selected_cols += [
            '1走前_order', '2走前_order', '3走前_order'
        ]
        # selected_cols += [col for col in df.columns if re.match(r"^[123]走前_", col)]
        # selected_cols += [col for col in df.columns if '近3走' in col]
        # selected_cols += [col for col in df.columns if '3R' in col]
        selected_cols += [col for col in df.columns if 'ランク_' in col]
        # ランク_人気_.*率 除外
        selected_cols = [col for col in selected_cols if not re.match(r'^ランク_人気_.*率$', col)]
        return selected_cols

    @staticmethod
    def create_data_core(df: pl.DataFrame, is_pred: bool, is_pandas: bool=False) -> Union[pl.DataFrame, pd.DataFrame]:
        """ 学習データを作成する
            Args:
                df pl.DataFrame: 元データ
                is_pred (bool): 予測用かどうか
                is_pandas (bool): pandas形式かどうか
            Returns:
                (pd.DataFrame or pl.DataFrame): 学習データ
        """
        selected_cols = Dataset.get_selected_columns(df)
        if not is_pred:
            selected_cols += ['order', 'top1','top2','top3','top4','top5','rank1','rank2','rank3','rank4','rank5']
            df = df.with_columns(
                pl.when(pl.col("order") == 1).then(16)
                .when(pl.col("order") == 2).then(8)
                .when(pl.col("order") == 3).then(4)
                .when(pl.col("order") == 4).then(2)
                .when(pl.col("order") == 5).then(1)
                .otherwise(0)
                .alias("rank5")
            )
            df = df.with_columns(
                pl.when(pl.col("order") == 1).then(16)
                .when(pl.col("order") == 2).then(8)
                .when(pl.col("order") == 3).then(4)
                .when(pl.col("order") == 4).then(2)
                .otherwise(0)
                .alias("rank4")
            )
            df = df.with_columns(
                pl.when(pl.col("order") == 1).then(16)
                .when(pl.col("order") == 2).then(8)
                .when(pl.col("order") == 3).then(4)
                .otherwise(0)
                .alias("rank3")
            )
            df = df.with_columns(
                pl.when(pl.col("order") == 1).then(16)
                .when(pl.col("order") == 2).then(8)
                .otherwise(0)
                .alias("rank2")
            )
            df = df.with_columns(
                pl.when(pl.col("order") == 1).then(16)
                .otherwise(0)
                .alias("rank1")
            )

        if is_pandas == False:
            return df.select(selected_cols)
        else:
            df = df.to_pandas()
            return df[selected_cols]

    @staticmethod
    def create_data(start:int, end:int, is_pred:bool=False, is_pandas:bool=False) -> None:
        """ 学習データを作成する
            Args:
                start(int): 年
                end(int): 年
                is_pred(bool): 予測用かどうか
            Returns:
                None
        """
        for year in tqdm(range(start, end)):
            df = pl.read_parquet(PathManager.get_rank_table_extra(year, False))
            df = Dataset.create_data_core(df, is_pred, is_pandas)

            path = PathManager.get_dataset_extra(year, False)
            dir_path = Path(path).parent
            if not os.path.isdir(dir_path):
                print(f"Creating folder {dir_path}")
                os.makedirs(dir_path)
            df.write_parquet(path)
            path = PathManager.get_dataset_extra(year)
            df = df.to_pandas()
            df.to_pickle(path)
    
    @staticmethod
    def read_dataset(start: int, end: int, is_pickle: bool = True) -> Union[pd.DataFrame, pl.DataFrame]:
        """ 学習データを取得する
            Args:
                start(int): 年
                end(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                pd.DataFrame: 学習データ
        """
        dfs = []
        for year in range(start, end):
            path = PathManager.get_dataset_extra(year, is_pickle)
            if os.path.isfile(path):
                if path.endswith(".parquet"):
                    df = pl.read_parquet(path)
                elif path.endswith(".pkl") or path.endswith(".pickle"):
                    df = pd.read_pickle(path)
                else:
                    raise ValueError(f"Unsupported file format: {path}")
                dfs.append(df)
            else:
                print(f"File not found: {path}")
        if is_pickle:
            df = pd.concat(dfs, axis=0, ignore_index=True)
        else:
            df = pl.concat(dfs, how='vertical')
        return df

    @staticmethod
    def delete_target(x: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        df = x.copy()
        #df.drop(['日付','調教日'], axis=1, inplace=True)
        for col in Dataset.DELETE_FEATURES:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        return df

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
        X_train_d = Dataset.delete_target(train_df)
        y_train_d = train_df[target]
        # 検証データの変換
        X_valid_d = Dataset.delete_target(valid_df)
        y_valid_d = valid_df[target]
        # テストデータの変換
        X_test_d = None
        y_test_d = None
        if test_len != 0:
            X_test_d = Dataset.delete_target(test_df)
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
    def extract_target_dataset_pd(df:pd.DataFrame, race_type:str='all', place:str='', distance:int=0, is_exclusion_new_horse:bool=True, distance_col:str='距離') -> pd.DataFrame:
        
        if is_exclusion_new_horse == True:
            temp = df[df['クラス'] != '新馬クラス']
        else:
            temp = df.copy()
    
        print(place, distance)
        if place != '':
            temp = temp[temp['会場'].isin([place])]
    
        if race_type == '芝':
            temp = temp[temp['レースタイプ']=='芝']
            if place == '':
                if distance == 0:
                    print(distance)
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1600]
                elif 2200 <= distance:
                    temp = temp[1800<=temp[distance_col]]
                else:
                    b = distance - 400
                    a = distance + 400
                    temp = temp[(b<=temp[distance_col])&(temp[distance_col]<=a)]
            elif place == '札幌':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1500]
                elif distance == 1500:
                    temp = temp[(1200<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1500<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif (distance == 2000) or (distance == 2600):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '函館':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1200]
                elif (distance == 1800) or (distance == 2000) or (distance == 2600):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '福島':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1200]
                elif distance == 1700:
                    temp = temp[(1700<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif (distance == 1800) or (distance == 2000) or (distance == 2600):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '新潟':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1400]
                elif distance == 1400:
                    temp = temp[(1000<=temp[distance_col])&(temp[distance_col]<=1600)]
                elif distance == 1600:
                    temp = temp[(1400<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif (distance == 2000) or (distance == 2200) or (distance == 2400) or (distance == 2850):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '東京':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1400:
                    temp = temp[temp[distance_col]<=1600]
                elif distance == 1600:
                    temp = temp[(1400<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2000:
                    temp = temp[(1800<=temp[distance_col])&(temp[distance_col]<=2400)]
                elif (distance == 2300) or (distance == 2400) or (distance == 2500) or (distance == 3400):
                    temp = temp[2000<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中山':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]<=1600]
                elif distance == 1600:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2000:
                    temp = temp[(1800<=temp[distance_col])&(temp[distance_col]<=2200)]
                elif (distance == 2200) or (distance == 2500) or (distance == 3600):
                    temp = temp[2000<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中京':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]<=1400]
                elif distance == 1400:
                    temp = temp[(1200<=temp[distance_col])&(temp[distance_col]<=1600)]
                elif distance == 1600:
                    temp = temp[(1400<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2000:
                    temp = temp[(1800<=temp[distance_col])&(temp[distance_col]<=2200)]
                elif (distance == 2200) or (distance == 2500) or (distance == 3000):
                    temp = temp[2000<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '京都':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]<=1400]
                elif distance == 1400:
                    temp = temp[(1200<=temp[distance_col])&(temp[distance_col]<=1600)]
                elif distance == 1600:
                    temp = temp[(1400<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2000:
                    temp = temp[(1800<=temp[distance_col])&(temp[distance_col]<=2200)]
                elif (distance == 2200) or (distance == 2400) or (distance == 3000) or (distance == 3200):
                    temp = temp[2000<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '阪神':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]<=1400]
                elif distance == 1400:
                    temp = temp[(1200<=temp[distance_col])&(temp[distance_col]<=1600)]
                elif distance == 1600:
                    temp = temp[(1400<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1600<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2000:
                    temp = temp[(1800<=temp[distance_col])&(temp[distance_col]<=2200)]
                elif (distance == 2200) or (distance == 2400) or (distance == 2600) or (distance == 3000) or (distance == 3200):
                    temp = temp[2000<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '小倉':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]==1200]
                elif distance == 1700:
                    temp = temp[(1700<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1700<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2000:
                    temp = temp[(1800<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif distance == 2600:
                    temp = temp[2000<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
        elif race_type == 'ダート':
            temp = temp[temp['レースタイプ']=='ダート']
            if place == '':
                if distance == 0:
                    print(distance)
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1600]
                elif 1800 <= distance:
                    temp = temp[1800<=temp[distance_col]]
                else:
                    b = distance - 400
                    a = distance + 400
                    temp = temp[(b<=temp[distance_col])&(temp[distance_col]<=a)]
            elif place == '札幌':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1700]
                elif (distance == 1700) or (distance == 2400):
                    temp = temp[1700<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '函館':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1700]
                elif (distance == 1700) or (distance == 2400):
                    temp = temp[1700<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '福島':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1200]
                elif (distance == 1700) or (distance == 2400):
                    temp = temp[1700<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '新潟':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1200]
                elif distance == 1700:
                    temp = temp[(1700<=temp[distance_col])&(temp[distance_col]<=2000)]
                elif (distance == 1800) or (distance == 2500):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '東京':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1300:
                    temp = temp[temp[distance_col]<=1400]
                elif distance == 1400:
                    temp = temp[temp[distance_col]==1400]
                elif distance == 1600:
                    temp = temp[temp[distance_col]==1600]
                elif (distance == 2100) or (distance == 2400):
                    temp = temp[1600<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中山':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]==1200]
                elif distance == 1800:
                    temp = temp[temp[distance_col]==1800]
                elif (distance == 2400) or (distance == 2500):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中京':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp[temp[distance_col]<=1400]
                elif distance == 1400:
                    temp = temp[(1200<=temp[distance_col])&(temp[distance_col]<=1400)]
                elif distance == 1700:
                    temp = temp[(1400<=temp[distance_col])&(temp[distance_col]<=1800)]
                elif distance == 1800:
                    temp = temp[(1700<=temp[distance_col])&(temp[distance_col]<=1900)]
                elif (distance == 1900) or (distance == 2300) or (distance == 2500):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '京都':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]==1200]
                elif distance == 1400:
                    temp = temp[temp[distance_col]==1400]
                elif (distance == 1800) or (distance == 1900):
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '阪神':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp[temp[distance_col]==1200]
                elif distance == 1400:
                    temp = temp[temp[distance_col]==1400]
                elif distance == 1800:
                    temp = temp[temp[distance_col]==1800]
                elif distance == 2000:
                    temp = temp[1800<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '小倉':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1000:
                    temp = temp[temp[distance_col]==1000]
                elif distance == 1700:
                    temp = temp[temp[distance_col]==1700]
                elif distance == 2400:
                    temp = temp[1700<=temp[distance_col]]
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
        
        print(f'race_ids = {len(temp.index.unique())}')
        return temp

    @staticmethod
    def extract_target_dataset_pl(df: pl.DataFrame, race_type:str='all', place:str='', distance:int=0, is_exclusion_new_horse:bool=True, distance_col:str='距離') -> pl.DataFrame:

        if is_exclusion_new_horse:
            temp = df.filter(pl.col('クラス') != '新馬クラス')
        else:
            temp = df

        if place != '':
            temp = temp.filter(pl.col('会場') == place)

        if race_type == '芝':
            temp = temp.filter(pl.col('レースタイプ') == '芝')
            if place == '':
                if distance == 0:
                    print(distance)
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1600)
                elif distance >= 2200:
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    temp = temp.filter((pl.col(distance_col) >= distance - 400) & (pl.col(distance_col) <= distance + 400))
            elif place == '札幌':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1500)
                elif distance == 1500:
                    temp = temp.filter((pl.col(distance_col) >= 1200) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1500) & (pl.col(distance_col) <= 2000))
                elif distance in [2000, 2600]:
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '函館':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1200)
                elif distance in [1800, 2000, 2600]:
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '福島':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1200)
                elif distance == 1700:
                    temp = temp.filter((pl.col(distance_col) >= 1700) & (pl.col(distance_col) <= 2000))
                elif (distance == 1800) or (distance == 2000) or (distance == 2600):
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '新潟':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1400)
                elif distance == 1400:
                    temp = temp.filter((pl.col(distance_col) >= 1000) & (pl.col(distance_col) <= 1600))
                elif distance == 1600:
                    temp = temp.filter((pl.col(distance_col) >= 1400) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 2200))
                elif (distance == 2000) or (distance == 2200) or (distance == 2400) or (distance == 2850):
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '東京':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1400:
                    temp = temp.filter(pl.col(distance_col) <= 1600)
                elif distance == 1600:
                    temp = temp.filter((pl.col(distance_col) >= 1400) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 2000))
                elif distance == 2000:
                    temp = temp.filter((pl.col(distance_col) >= 1800) & (pl.col(distance_col) <= 2400))
                elif (distance == 2300) or (distance == 2400) or (distance == 2500) or (distance == 3400):
                    temp = temp.filter(pl.col(distance_col) >= 2000)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '中山':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1600)
                elif distance == 1600:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 2000))
                elif distance == 2000:
                    temp = temp.filter((pl.col(distance_col) >= 1800) & (pl.col(distance_col) <= 2200))
                elif (distance == 2200) or (distance == 2500) or (distance == 3600):
                    temp = temp.filter(pl.col(distance_col) >= 2000)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中京':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1400)
                elif distance == 1400:
                    temp = temp.filter((pl.col(distance_col) >= 1200) & (pl.col(distance_col) <= 1600))
                elif distance == 1600:
                    temp = temp.filter((pl.col(distance_col) >= 1400) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 2000))
                elif distance == 2000:
                    temp = temp.filter((pl.col(distance_col) >= 1800) & (pl.col(distance_col) <= 2200))
                elif (distance == 2200) or (distance == 2500) or (distance == 3000):
                    temp = temp.filter(pl.col(distance_col) >= 2000)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '京都':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1400)
                elif distance == 1400:
                    temp = temp.filter((pl.col(distance_col) >= 1200) & (pl.col(distance_col) <= 1600))
                elif distance == 1600:
                    temp = temp.filter((pl.col(distance_col) >= 1400) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 2000))
                elif distance == 2000:
                    temp = temp.filter((pl.col(distance_col) >= 1800) & (pl.col(distance_col) <= 2200))
                elif (distance == 2200) or (distance == 2400) or (distance == 3000) or (distance == 3200):
                    temp = temp.filter(pl.col(distance_col) >= 2000)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '阪神':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1400)
                elif distance == 1400:
                    temp = temp.filter((pl.col(distance_col) >= 1200) & (pl.col(distance_col) <= 1600))
                elif distance == 1600:
                    temp = temp.filter((pl.col(distance_col) >= 1400) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1600) & (pl.col(distance_col) <= 2000))
                elif distance == 2000:
                    temp = temp.filter((pl.col(distance_col) >= 1800) & (pl.col(distance_col) <= 2200))
                elif (distance == 2200) or (distance == 2400) or (distance == 2600) or (distance == 3000) or (distance == 3200):
                    temp = temp.filter(pl.col(distance_col) >= 2000)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '小倉':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) == 1200)
                elif distance == 1700:
                    temp = temp.filter((pl.col(distance_col) >= 1700) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1700) & (pl.col(distance_col) <= 2000))
                elif distance == 2000:
                    temp = temp.filter((pl.col(distance_col) >= 1800) & (pl.col(distance_col) <= 2000))
                elif distance == 2600:
                    temp = temp.filter(pl.col(distance_col) >= 2000)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
        elif race_type == 'ダート':
            temp = temp.filter(pl.col('レースタイプ') == 'ダート')
            if place == '':
                if distance == 0:
                    print(distance)
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1600)
                elif distance >= 1800:
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    temp = temp.filter((pl.col(distance_col) >= distance - 400) & (pl.col(distance_col) <= distance + 400))
            elif place == '札幌':
                if distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1700)
                elif distance in [1700, 2400]:
                    temp = temp.filter(pl.col(distance_col) >= 1700)
                else:
                    raise Exception(f'{place} distance not defined {distance}')
            elif place == '函館':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1700)
                elif (distance == 1700) or (distance == 2400):
                    temp = temp.filter(pl.col(distance_col) >= 1700)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '福島':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1200)
                elif (distance == 1700) or (distance == 2400):
                    temp = temp.filter(pl.col(distance_col) >= 1700)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '新潟':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1200)
                elif distance == 1700:
                    temp = temp.filter((pl.col(distance_col) >= 1700) & (pl.col(distance_col) <= 2000))
                elif (distance == 1800) or (distance == 2500):
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '東京':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1300:
                    temp = temp.filter(pl.col(distance_col) <= 1400)
                elif distance == 1400:
                    temp = temp.filter(pl.col(distance_col) == 1400)
                elif distance == 1600:
                    temp = temp.filter(pl.col(distance_col) == 1600)
                elif (distance == 2100) or (distance == 2400):
                    temp = temp.filter(pl.col(distance_col) >= 1600)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中山':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) == 1200)
                elif distance == 1800:
                    temp = temp.filter(pl.col(distance_col) == 1800)
                elif (distance == 2400) or (distance == 2500):
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '中京':
                if distance == 0:
                    print('distance == 0')
                elif distance <= 1200:
                    temp = temp.filter(pl.col(distance_col) <= 1400)
                elif distance == 1400:
                    temp = temp.filter((pl.col(distance_col) >= 1200) & (pl.col(distance_col) <= 1400))
                elif distance == 1700:
                    temp = temp.filter((pl.col(distance_col) >= 1400) & (pl.col(distance_col) <= 1800))
                elif distance == 1800:
                    temp = temp.filter((pl.col(distance_col) >= 1700) & (pl.col(distance_col) <= 1900))
                elif (distance == 1900) or (distance == 2300) or (distance == 2500):
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '京都':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) == 1200)
                elif distance == 1400:
                    temp = temp.filter(pl.col(distance_col) == 1400)
                elif (distance == 1800) or (distance == 1900):
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '阪神':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1200:
                    temp = temp.filter(pl.col(distance_col) == 1200)
                elif distance == 1400:
                    temp = temp.filter(pl.col(distance_col) == 1400)
                elif distance == 1800:
                    temp = temp.filter(pl.col(distance_col) == 1800)
                elif distance == 2000:
                    temp = temp.filter(pl.col(distance_col) >= 1800)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 
            elif place == '小倉':
                if distance == 0:
                    print('distance == 0')
                elif distance == 1000:
                    temp = temp.filter(pl.col(distance_col) == 1000)
                elif distance == 1700:
                    temp = temp.filter(pl.col(distance_col) == 1700)
                elif distance == 2400:
                    temp = temp.filter(pl.col(distance_col) >= 1700)
                else:
                    raise Exception(f'{place} distance not defined {distance}') 

        print(f"race_ids = {temp.select('race_id').unique().height}")
        return temp

    @staticmethod
    def convert_column_type_pd(src_df: pd.DataFrame, is_cnv_category:bool=True, is_pred:bool=False) -> pd.DataFrame:
        df = src_df.copy()
        for day_col in ["日付", "調教日"]:
            if day_col in df.columns and df[day_col].dtype != 'datetime64[ns]':
                df[day_col] = pd.to_datetime(df[day_col])

        bool_cols = df.select_dtypes(include='bool').columns
        if len(bool_cols) != 0:
            df[bool_cols] = df[bool_cols].astype('int')
            
        if is_cnv_category == True:
            cols = df.select_dtypes(include='object').columns
            if len(cols) != 0:
                df[cols] = df[cols].astype('category')
        else:
            cols = df.select_dtypes(include='category').columns
            if len(cols) != 0:
                df[cols] = df[cols].astype('object')

        if is_pred:
            df = df.drop(columns=['日付'])
            for col in ['order', 'top1', 'top2', 'top3', 'top4', 'top5', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5']:
                if col in df.columns:
                    df = df.drop(columns=[col])
        return df

    @staticmethod
    def convert_column_type_pl(src_df: pl.DataFrame, is_cnv_category: bool = True, is_conv_pandas:bool=True) -> Union[pd.DataFrame, pl.DataFrame]:
        """ データ型を変換する
            Args:
                src_df (pl.DataFrame): 元データ
                is_cnv_category (bool): category に変換するかどうか
                is_conv_pandas (bool): pandas に変換するかどうか
            Returns:
                pl.DataFrame: 変換後のデータ
        """
        df = src_df.clone()

        # 日付列を Datetime 型に変換
        for date_col in ["日付", "調教日"]:
            if date_col in df.columns and df[date_col].dtype not in [pl.Date, pl.Datetime]:
                df = df.with_columns(
                    pl.col(date_col).str.strptime(pl.Date, fmt="%Y-%m-%d", strict=False)
                )

        # bool型を int に変換（Polarsではbool→intは明示的に変換する）
        for col in df.columns:
            if df[col].dtype == pl.Boolean:
                df = df.with_columns(
                    pl.col(col).cast(pl.Int8)
                )

        # object型（=String）を category 的に扱いたい場合
        if is_cnv_category:
            for col in df.columns:
                if df[col].dtype == pl.Utf8:
                    # category 的に扱うには 'categorical' に変換
                    df = df.with_columns(
                        pl.col(col).cast(pl.Categorical)
                    )
        else:
            # categorical → string に戻す
            for col in df.columns:
                if df[col].dtype == pl.Categorical:
                    df = df.with_columns(
                        pl.col(col).cast(pl.Utf8)
                    )

        if is_conv_pandas:
            # Polars → Pandas に変換
            df = df.to_pandas()
            # datetime 型を datetime64[ns] に変換
            for date_col in ["日付", "調教日"]:
                if date_col in df.columns and df[date_col].dtype != np.datetime64:
                    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")

        return df

    @staticmethod
    def convert_distance(race_type: str, place: str, dis: int) -> int:
        """ 距離の変換
            Args:
                race_type (str): レースの種類 ('芝' or 'ダート')
                place (str): 開催場所 ('札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉')
                dis (int): 距離
            Returns:
                int: 変換後の距離
        """
        distance = dis
        if race_type == '芝':
            if place == '札幌':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 2000) or (distance == 2600):
                    distance = 2000
            elif place == '函館':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1800) or (distance == 2000) or (distance == 2600):
                    distance = 1800
            elif place == '福島':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1800) or (distance == 2000) or (distance == 2600):
                    distance = 1800
            elif place == '新潟':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 2000) or (distance == 2200) or (distance == 2400) or (distance == 2850):
                    distance = 2000
            elif place == '東京':
                if (distance == 2300) or (distance == 2400) or (distance == 2500) or (distance == 3400):
                    distance = 2400
            elif place == '中山':
                if (distance == 2200) or (distance == 2500) or (distance == 3600):
                    distance = 2200
            elif place == '中京':
                if (distance == 2200) or (distance == 2500) or (distance == 3000):
                    distance = 2200
            elif place == '京都':
                if (distance == 2200) or (distance == 2400) or (distance == 3000) or (distance == 3200):
                    distance = 2200
            elif place == '阪神':
                if (distance == 2200) or (distance == 2400) or (distance == 2600) or (distance == 3000) or (distance == 3200):
                    distance = 2200
            elif place == '小倉':
                pass
        elif race_type == 'ダート':
            if place == '札幌':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1700) or (distance == 2400):
                    distance = 1700
            elif place == '函館':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1700) or (distance == 2400):
                    distance = 1700
            elif place == '福島':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1700) or (distance == 2400):
                    distance = 1700
            elif place == '新潟':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1800) or (distance == 2500):
                    distance = 1800
            elif place == '東京':
                if (distance == 2100) or (distance == 2400):
                    distance = 2100
            elif place == '中山':
                if (distance == 2400) or (distance == 2500):
                    distance = 2400
            elif place == '中京':
                if distance <= 1200:
                    distance = 1200
                elif (distance == 1900) or (distance == 2300) or (distance == 2500):
                    distance = 1900
            elif place == '京都':
                if (distance == 1800) or (distance == 1900):
                    distance = 1800
            elif place == '阪神':
                pass 
            elif place == '小倉':
                pass
        return distance

    @staticmethod
    def create_dataset(days: list, start_year: int, end_year: int) -> None:
        """ データセット作成関数
            Args:
                days (list): 日付のリスト
                start_year (int): 開始年
                end_year (int): 終了年
            Returns:
                None
        """
        Notification.send('学習データ作成 開始')
        try:
            for weekend in days:
                print('race_id スクレイピング')
                NetKeibaAccesser.run(RaceId.scraping, days=weekend)
                print('race_id データ作成')
                RaceId.create_data(start_year, end_year)
                print('出馬表 スクレイピング')
                NetKeibaAccesser.run(RaceTable.scraping, days=weekend)
                print('出馬表 データ作成')
                RaceTable.create_data(start_year, end_year)
                print('払戻表 データ作成')
                RefundTable.create_data(start_year, end_year)
                print('血統表 スクレイピング')
                NetKeibaAccesser.run(PedigreeTable.scraping, days=weekend)
                print('血統表 データ作成')
                PedigreeTable.create_data()
                print('調教 スクレイピング')
                NetKeibaAccesser.run(TrainingTable.scraping, days=weekend)
                print('調教 データ作成')
                TrainingTable.create_data(start_year, end_year)
                print('出馬表 前処理')
                RaceTable.preprocess(start_year, end_year)
                print('血統表 前処理')
                PedigreeTable.preprocess()
                print('調教 前処理')
                TrainingTable.preprocess(start_year, end_year)
                print('前処理 マージ')
                RaceTable.merge_preprocess_data(start_year, end_year)
                
            print('ホースレート データ作成')
            HorseRateData.create_data(end_year)
            print('過去データ作成')
            PastData.create_past_data_extra(start_year, end_year)
            print('統計データ作成')
            StaticticsData.create_statictics_data_extra(start_year, end_year)
            print('ランクデータ作成')
            RankingData.create_rank_data_extra(start_year, end_year)
            print('データセット作成')
            Dataset.create_data(start_year, end_year)
        except Exception as e:
            Notification.send(f'学習データ: {e}')
        else:
            Notification.send('学習データ作成 完了')
