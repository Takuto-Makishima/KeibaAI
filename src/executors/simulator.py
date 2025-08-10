import os
import pandas as pd
import polars as pl
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from tqdm import tqdm

from typing import List, Union
from src.creators.dataset import Dataset
from src.executors.forecaster import Forecaster
from src.executors.today_data import TodayData
from src.table_data.race_table import RaceTable
from src.table_data.pedigree_table import PedigreeTable
from src.table_data.training_table import TrainingTable
from src.table_data.refund_table import RefundTable
from src.utils.netkeiba_accesser import NetKeibaAccesser
from src.utils.notification import Notification

class Simulator:
    """ シミュレータークラス """
    @staticmethod
    def read_predict_result(days: list, objective: str, target_value: str, target_name: str='predict_df')-> pd.DataFrame:
        """ 予測結果読み込み関数
            Args:
                days (list): 日付のリスト
                objective (str): 目的関数名
                target_value (str): ターゲット値
                target_name (str): ターゲット名
            Returns:
                pd.DataFrame: 読み込んだ予測結果のデータフレーム
        """
        result_df = pd.DataFrame()
        for date in tqdm(days):
            print(date[0], date[-1])
            for today in date:
                print(today)
        
                str_day = datetime.strftime(today, '%Y%m%d')
                file_path = f'./html/predict/{str_day}/{objective}/{target_value}/{target_name}.pickle'
                if os.path.isfile(file_path) == False:
                    print(f'continue {str_day}')
                    continue
                read_df = pd.read_pickle(file_path)
                result_df = pd.concat([result_df, read_df])

        return result_df

    @staticmethod
    def read_refund_data(days: list) -> RefundTable:
        """ 払戻データ読み込み関数
            Args:
                days (list): 日付のリスト
            Returns:
                RefundTable: 払戻表のデータフレーム
        """
        refund_df = pd.DataFrame()

        for date in tqdm(days):
            print(date[0], date[-1])
            for today in date:
                print(today)
                str_day = datetime.strftime(today, '%Y%m%d')
                file_path = f'./html/predict/{str_day}/result_{str_day}.pickle'
                if os.path.isfile(file_path) == False:
                    print(f'continue {str_day}')
                    continue
                refund_df = pd.concat([refund_df, pd.read_pickle(file_path)])

        refund_df["race_id"] = refund_df.index
        refund_table_df = RefundTable(pl.from_pandas(refund_df), is_today=True)

        return refund_table_df

    @staticmethod
    def create_summary_table(result_df: pd.DataFrame, count: int, exists_past_3rd: bool, exists_past_2nd: bool,
                             threshold: float = 0.0, is_today: bool=False, is_expected: bool=False,
                             target_col: str='pred', is_sort: bool=True) -> pd.DataFrame:
        """ 集計用テーブル作成
            Args:
                result_df (pd.DataFrame): 予測結果のデータフレーム
                count (int): 予測数
                exists_past_3rd (bool): 過去3走の着順が存在するかどうか
                exists_past_2nd (bool): 過去2走の着順が存在するかどうか
                threshold (float): 予測値の閾値
                is_today (bool): 今日のレースのみを対象とするかどうか
                is_expected (bool): 期待値を計算するかどうか
                target_col (str): 対象列名
                is_sort (bool): ソートするかどうか
            Returns:
                pd.DataFrame: 予測テーブル
        """
        columns = []
        columns.append('日付')
        columns.append('レースタイプ')        
        columns.append('会場')
        columns.append('距離')
        columns.append('馬場')

        for i in range(0, count):
            columns.append(f'p_f_{i+1}')
            columns.append(f'p_h_{i+1}')
            columns.append(f'p_p_{i+1}')
            columns.append(f'pred_{i+1}')

        race_ids = result_df["race_id"].unique()
        tbl_pred = pd.DataFrame(data=[], index=race_ids, columns=columns)

        day = '日付'
        place = '会場'
        race_type = 'レースタイプ'
        distance = '距離'
        grade = 'クラス'
        frame_num = '枠番'
        horse_num = '馬番'
        popularity = '人気'
        ground = '馬場'

        for race_id in tqdm(race_ids):
            target_df = result_df.query("race_id == @race_id").copy()#loc[result_df["race_id"]==race_id].copy()
            target_day = target_df[day].iloc[0]
            target_place = target_df[place].iloc[0]
            target_type = target_df[race_type].iloc[0]
            target_distance = target_df[distance].iloc[0]
            target_class = target_df[grade].iloc[0]
            target_ground = target_df[ground].iloc[0]
            
            if target_df[target_col].isna().any():
                print('continue', race_id, target_place, target_type[0], int(target_distance), target_class)
                continue

            if exists_past_3rd == True:
                past_3rd = target_df['3走前_着順'].notna().all()
                if past_3rd == False:
                    print('continue', race_id, target_place, target_type[0], int(target_distance), target_class)
                    continue

            if exists_past_2nd == True:
                past_2nd = target_df['2走前_着順'].notna().all()
                if past_2nd == False:
                    print('continue', race_id, target_place, target_type[0], int(target_distance), target_class)
                    continue

            disp_df = None
            if is_expected == True:
                target_df['標準化'] = target_df[target_col].transform(lambda x: ((x - x.mean()) / x.std()))
                target_df['偏差値'] = target_df[target_col].transform(lambda x: ((x - x.mean()) / x.std()) * 10 + 50)
                target_df['期待値'] = target_df['偏差値'] * target_df['単勝']

            if is_sort==True:
                disp_df = target_df.sort_values(target_col, ascending=False)
            else:
                disp_df = target_df.copy()

            # 指定列に値が入っていることを確認
            # count_above_zero = disp_df[target_col].apply(lambda x: x > threshold).sum()
            # is_pred_col = False
            # if count_above_zero < count:
            #     print(f'continue under count {count_above_zero} < {count}', race_id, target_place, target_type[0], int(target_distance), target_class)
            #     continue

            # 上位取得
            temp = disp_df.head(5).copy()
            value = temp[target_col].iloc[-1]
            cnt = 1
            for index, frame, horse, prd, dis, ev_id in zip(temp["race_id"], temp[frame_num], temp[horse_num], temp[target_col], temp[distance], temp[place]):
                tbl_pred.loc[index,'日付'] = target_day
                tbl_pred.loc[index,'レースタイプ'] = target_type
                tbl_pred.loc[index,'会場'] = target_place
                tbl_pred.loc[index,'距離'] = target_distance
                tbl_pred.loc[index,'馬場'] = target_ground
                tbl_pred.loc[index,f'p_f_{cnt}'] = frame
                tbl_pred.loc[index,f'p_h_{cnt}'] = horse
                tbl_pred.loc[index,f'p_p_{cnt}'] = 0
                tbl_pred.loc[index,f'pred_{cnt}'] = prd
                cnt += 1

        return tbl_pred

    @staticmethod
    def add_place_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 単勝結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_place'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                if np.isnan(horse_1) == True:
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    if (horse_1 == int(win)):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_win_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 単勝結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_win'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                if np.isnan(horse_1) == True:
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    if (horse_1 == int(win)):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_wide_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ ワイド結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_wide'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                if (np.isnan(float(horse_1)) == True) or (np.isnan(float(horse_2)) == True):
                    continue
                # win列取得
                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)           
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']
                    if (horse_1 in new_list) and (horse_2 in new_list):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_bracket_quinella_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 枠連結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_bracket_quinella'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_f_1'])
                horse_2 = float(pred.loc[idx, 'p_f_2'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True):
                    continue
                # win列取得
                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)           
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']
                    if ((horse_1 == new_list[0]) and (horse_2 == new_list[1])) or ((horse_1 == new_list[1]) and (horse_2 == new_list[0])):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_quinella_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 馬連結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_quinella'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']
                    if ((horse_1 == new_list[0]) and (horse_2 == new_list[1])) or ((horse_1 == new_list[1]) and (horse_2 == new_list[0])):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_exacta_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 馬単結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        taeget = 'refund_exacta'
        pred[taeget] = np.nan
        for idx in pred.index:
            # 単勝
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']                   
                    if (new_list[0] == horse_1) and (new_list[1] == horse_2):
                        pred.loc[idx,taeget] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,taeget] = -100

        return pred

    @staticmethod
    def add_trio_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 三連複結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_trio'
        pred[target] = np.nan
        for idx in pred.index:
            # 単勝
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                horse_3 = float(pred.loc[idx, 'p_h_3'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True) or (np.isnan(horse_3) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                horse_3 = int(horse_3)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']                   
                    if (horse_1 in new_list) and (horse_2 in new_list) and (horse_3 in new_list):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100

        return pred

    @staticmethod
    def add_tierce_result(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame) -> pd.DataFrame:
        """ 三連単結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        taeget = 'refund_tierce'
        pred[taeget] = np.nan
        for idx in pred.index:
            # 単勝
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                horse_3 = float(pred.loc[idx, 'p_h_3'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True) or (np.isnan(horse_3) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                horse_3 = int(horse_3)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']                   
                    if (new_list[0] == horse_1) and (new_list[1] == horse_2) and (new_list[2] == horse_3):
                        pred.loc[idx,taeget] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,taeget] = -100

        return pred

    @staticmethod
    def calculation_data(tbl: pd.DataFrame) -> pd.DataFrame:
        """ 払戻データの集計
            Args:
                tbl (pd.DataFrame): 払戻テーブル
            Returns:
                pd.DataFrame: 集計結果
        """
        lst = ['refund_place','refund_win','refund_wide','refund_bracket_quinella','refund_quinella','refund_exacta','refund_trio','refund_tierce']
        result = None
        for col in lst:
            tbl_ref = tbl.copy()
            tbl_ref = tbl_ref[tbl_ref[col].notna()].copy()
            tbl_ref[col] = tbl_ref[col].astype(int)
            # 総レース数
            total_count = len(tbl)
            #　購入数
            buy = len(tbl_ref[tbl_ref[col].notna()])
            # 購入率
            buy_rate = buy / total_count
            # 的中
            hits = tbl_ref[tbl_ref[col] != -100][col]
            # 的中数(払戻数)
            hit_count = len(hits)
            # 的中率
            hit_rate = hit_count / buy
            # 購入金額
            purchase_price = buy * 100
            # 払戻金額
            refund = (hits + 100).sum()
            # 収支
            total = tbl_ref[col].sum()
            # 回収率
            recovery_rate = refund / purchase_price

            lst = [f'{total_count}R', f'{buy}R', f'{buy_rate*100:.1f}%', f'{hit_count}', f'{hit_rate*100:.1f}%',f'{purchase_price}円', f'{refund}円', f'{total}円', f'{recovery_rate*100:.1f}%']

            df = pd.DataFrame(lst, index =['総レース数', '購入数', '購入率', '的中数', '的中率', '購入金額', '払戻金額', '収支', '回収率'],columns =[col.replace('refund_', '')])
            if type(result) != None:
                result = pd.concat([result, df], axis=1)
            else:
                result = df
        return result

    @staticmethod
    def show_graph(pred_tbl: pd.DataFrame, lst=['refund_place','refund_win','refund_wide','refund_bracket_quinella','refund_quinella','refund_exacta','refund_trio','refund_tierce']):
        """ 予測結果のグラフを表示
            Args:
                pred_tbl (pd.DataFrame): 予測テーブル
                lst (list): 表示する列のリスト
            Returns:
                None: グラフを表示する
        """
        plt.figure(figsize=(15,20))
        row = 4
        col  = 2
        cnt = 1
        unique_dates = pred_tbl['日付'].dropna().sort_values().unique()
        colors = list(mcolors.TABLEAU_COLORS.values())
        pred = pred_tbl.copy()
        print('プロット')
        for columns in tqdm(lst):
            pred = pred[pred[columns].notna()]
            plt.subplot(row, col, cnt, title=columns.replace('refund_',''))
            plt.grid(axis='y')
            pred[f'{columns}_sum'] = pred[columns].cumsum()
            plt.plot(pred[f'{columns}_sum'])
            cnt = cnt + 1
            # 背景に日付ごとの範囲を塗る
            for i, date in enumerate(unique_dates):
                date_data = pred[pred['日付'] == date]
            
                start_idx = date_data.index[0]
                end_idx = date_data.index[-1]
                plt.axvspan(start_idx, end_idx, color=colors[i % len(colors)], alpha=0.3)
        #グラフを表示
        plt.show()

    @staticmethod
    def process_combo_box_bet(tbl_pred: pd.DataFrame, disp_df: pd.DataFrame, refund_table: pd.DataFrame, target: str, targets: list, combo: int) -> tuple:
        """ 連複系の集計結果をまとめる
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                disp_df (pd.DataFrame): 表示用データフレーム
                refund_table (pd.DataFrame): 払戻テーブル
                target (str): ターゲット列名
                targets (list): 対象馬のリスト
                combo (int): コンボ数
            Returns:
                pd.DataFrame, pd.DataFrame: 更新された予測テーブルと表示用データフレーム
        """
        pattern = sum(1 for _ in itertools.combinations(targets, combo))
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table, targets, combo)
        tbl_pred[target] = tbl[target]
        calc_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, calc_df], axis=1)
        tbl_pred[target] = tbl_pred[target] - (pattern * 100)
        return tbl_pred, disp_df

    @staticmethod
    def process_prem_box_bet(tbl_pred: pd.DataFrame, disp_df: pd.DataFrame, refund_table: pd.DataFrame, target: str, targets: list, combo: int) -> tuple:
        """ 連単系の集計結果をまとめる
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                disp_df (pd.DataFrame): 表示用データフレーム
                refund_table (pd.DataFrame): 払戻テーブル
                target (str): ターゲット列名
                targets (list): 対象馬のリスト
                combo (int): コンボ数
            Returns:
                pd.DataFrame, pd.DataFrame: 更新された予測テーブルと表示用データフレーム
        """
        pattern = sum(1 for _ in itertools.permutations(targets, combo))
        tbl = Simulator.add_perm_box(tbl_pred, target, refund_table, targets, combo)
        tbl_pred[target] = tbl[target]
        calc_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, calc_df], axis=1)
        tbl_pred[target] = tbl_pred[target] - (pattern * 100)
        return tbl_pred, disp_df

    @staticmethod
    def add_combo_box(tbl_pred: pd.DataFrame, target: str, table: pd.DataFrame, pattern: list, ext: int) -> pd.DataFrame:
        """ 連複系のボックス購入時の結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                target (str): ターゲット列名
                table (pd.DataFrame): 払戻テーブル
                pattern (list): パターンリスト
                ext (int): 拡張数
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = target
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        for idx, row in tbl_pred[tbl_pred['p_h_1'].notna()][pattern].iterrows():
            # idx 存在確認
            if idx not in table.index:
                continue
            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(table.filter(like='win').columns)
            for i in range(0, cnt):
                win = table.loc[idx, f'win_{i}']
                if (type(win) == type(None)) or (win == 0):
                    continue

                if type(win) == str:
                    win = win.split(' ')
                    win = [x for x in win if x != '']
                    win = set([int(x) for x in win])

                for item in itertools.combinations(row, ext):
                    if isinstance(win,np.int32) or isinstance(win,np.int64):
                        item = int(item[0])
                    else:
                        item = set([int(x) for x in item])

                    if item == win:
                        #print(idx, item, win)
                        pred.loc[idx,target] += (int(table.loc[idx,f'ref_{i}']))
                        pred.loc[idx,f'{target}_hit'] = True
                        break

        return pred

    @staticmethod
    def lists_match(l1: list, l2: list) -> bool:
        """ リストの要素が一致するか確認
            Args:
                l1 (list): リスト1
                l2 (list): リスト2
            Returns:
                bool: 一致する場合はTrue, それ以外はFalse
        """
        if len(l1) != len(l2):
            return False
        return all(x == y and type(x) == type(y) for x, y in zip(l1, l2))

    @staticmethod
    def add_perm_box(tbl_pred: pd.DataFrame, target: str, table: pd.DataFrame, pattern: list, ext: int) -> pd.DataFrame:
        """ 連単系のボックス購入時の結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                target (str): ターゲット列名
                table (pd.DataFrame): 払戻テーブル
                pattern (list): パターンリスト
                ext (int): 拡張数
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = target
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        for idx, row in tbl_pred[tbl_pred['p_h_1'].notna()][pattern].iterrows():
            # idx 存在確認
            if idx not in table.index:
                continue

            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(table.filter(like='win').columns)
            for i in range(0, cnt):
                win = table.loc[idx, f'win_{i}']
                if (type(win) == type(None)) or (win == 0):
                    continue

                win = win.split(' ')
                win = [x for x in win if x != '']
                win = [int(x) for x in win]

                for item in itertools.permutations(row, ext):
                    item = [int(x) for x in item]
                    if Simulator.lists_match(win, item):
                        #print(idx, item, win)
                        pred.loc[idx,target] += (int(table.loc[idx,f'ref_{i}']))
                        pred.loc[idx,f'{target}_hit']=True
                        break

        return pred

    @staticmethod
    def calcuration(table: pd.DataFrame, col: str, count: int) -> pd.DataFrame:
        """ 払戻データの集計 
            Args:
                table (pd.DataFrame): 払戻テーブル
                col (str): 集計対象の列名
                count (int): 購入数
            Returns:
                pd.DataFrame: 集計結果
        """
        tbl = table.copy()
        # 総レース数
        total_count = len(tbl)

        tbl = tbl[tbl[col].notna()].copy()
        tbl[col] = tbl[col].astype(int)

        #　購入数
        buy = len(tbl)
        # 購入率
        buy_rate = buy / total_count

        # 的中
        hits = tbl[tbl[f'{col}_hit'] == True][col]
        # 的中数(払戻数)
        hit_count = len(hits)
        # 的中率
        hit_rate = hit_count / buy

        # 購入金額
        purchase_price = buy * (100 * count)
        # 払戻金額
        refund = int(hits.sum())
        # 収支
        total = refund - purchase_price
        # 回収率
        recovery_rate = refund / purchase_price

        lst = [f'{total_count}R', f'{buy}R', f'{buy_rate*100:.1f}%', f'{hit_count}', f'{hit_rate*100:.1f}%',f'{purchase_price}円', f'{refund}円', f'{total}円', f'{recovery_rate*100:.1f}%']

        df = pd.DataFrame(lst, index =['総レース数', '購入数', '購入率', '的中数', '的中率', '購入金額', '払戻金額', '収支', '回収率'],columns =[col.replace('refund_', '')])
            
        return df

    @staticmethod
    def process_formation_bet(tbl_pred: pd.DataFrame, disp_df: pd.DataFrame, refund_table: pd.DataFrame, target: str, first: list, second: list, third: list=None) -> tuple:
        """ フォーメーション系の集計結果をまとめる
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                disp_df (pd.DataFrame): 表示用データフレーム
                refund_table (pd.DataFrame): 払戻テーブル
                target (str): ターゲット列名
                first (list): 第一対象馬のリスト
                second (list): 第二対象馬のリスト
                third (list, optional): 第三対象馬のリスト
            Returns:
                pd.DataFrame, pd.DataFrame: 更新された予測テーブルと表示用データフレーム
        """
        if third is None:
            pattern = len(first) * (len(second) - 1)
            tbl = Simulator.add_exacta_formation(tbl_pred, refund_table, [first, second], pattern)
        else:
            pattern = len(first) * (len(second) - 1) * (len(third) - 2)
            tbl = Simulator.add_tierce_formation(tbl_pred, refund_table, [first, second, third], pattern)
        tbl_pred[target] = tbl[target]
        calc_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, calc_df], axis=1)
        tbl_pred[target] = tbl_pred[target] - (pattern * 100)
        return tbl_pred, disp_df

    @staticmethod
    def add_exacta_formation(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame, lst: list, pattern: int) -> pd.DataFrame:
        """ 馬単フォーメーション結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
                lst (list): 対象列のリスト
                pattern (int): フォーメーションのパターン
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_exacta_formation'
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        # 予測テーブルループ
        for idx in pred.index:
            # idx 存在確認
            if idx not in refund_table.index:
                continue
            if pd.isna(tbl_pred.loc[idx, 'p_h_1']):
                continue

            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(refund_table.filter(like='win').columns)
            for num in range(0, cnt):
                win = refund_table.loc[idx, f'win_{num}']
                if (type(win) == type(None)) or (pd.isnull(pred.loc[idx, lst[0]].iloc[0])):
                    continue
                win = win.split(' ')
                win = [x for x in win if x != '']
                if (int(win[0]) in pred.loc[idx, lst[0]].to_list()) and (int(win[1]) in pred.loc[idx, lst[1]].to_list()):
                    pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{num}']))
                    pred.loc[idx,f'{target}_hit']=True

        return pred

    @staticmethod
    def add_tierce_formation(tbl_pred: pd.DataFrame, refund_table: pd.DataFrame, lst: list, pattern: int) -> pd.DataFrame:
        """ 三連単フォーメーション結果を予測テーブルに追加
            Args:
                tbl_pred (pd.DataFrame): 予測テーブル
                refund_table (pd.DataFrame): 払戻テーブル
                lst (list): 対象列のリスト
                pattern (int): フォーメーションのパターン
            Returns:
                pd.DataFrame: 更新された予測テーブル
        """
        pred = tbl_pred.copy()
        target = 'refund_tierce_formation'
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        # 予測テーブルループ
        for idx in pred.index:
            # idx 存在確認
            if idx not in refund_table.index:
                continue
            if pd.isna(tbl_pred.loc[idx, 'p_h_1']):
                continue

            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(refund_table.filter(like='win').columns)
            for num in range(0, cnt):
                win = refund_table.loc[idx, f'win_{num}']
                if (type(win) == type(None)) or (pd.isnull(pred.loc[idx, lst[0]].iloc[0])):
                    continue
                win = win.split(' ')
                win = [x for x in win if x != '']
                if (int(win[0]) in pred.loc[idx, lst[0]].to_list()) and (int(win[1]) in pred.loc[idx, lst[1]].to_list()) and (int(win[2]) in pred.loc[idx, lst[2]].to_list()):
                    pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{num}']))
                    pred.loc[idx,f'{target}_hit']=True

        return pred

    @staticmethod
    def create_prediction_data(days: list, start_year: int, end_year: int, is_past: bool, objective: str) -> None:
        """ シミュレーション作成関数
            Args:
                days (list): 日付のリスト
                start_year (int): 開始年
                end_year (int): 終了年
                is_past (bool): 過去データを使用するかどうか
                objective (str): 目的関数名
            Returns:
                None
        """
        Notification.send('シミュレーション作成 開始')
        target_df = None
        try:
            print('シミュレーション作成')
            for date in tqdm(days):
                print(date[0], date[-1])
                for today in date:
                    print(f'日付: {today}')
                    main = Forecaster(today, date, objective)
                    main.initialize()

                    str_day = datetime.strftime(main.today, '%Y%m%d')
                    result_df = pd.DataFrame()
                    for race_id in tqdm(main.time_table[main.time_table['start_time'].notna()].index):
                        if race_id == 'win5':
                            continue

                        start_time = main.time_table.loc[race_id, 'start_time']
                        dir_path = f'./html/predict/{str_day}/{race_id}'
                        file_path = f'{dir_path}/predict_dataset.pickle'
                        print(dir_path, start_time)

                        if is_past:
                            if target_df is None:
                                target_df = Dataset.read_dataset(start_year, end_year, True)
                            predict_df = target_df[target_df["race_id"] == race_id].copy()
                            if '障' in predict_df.iloc[0]["レースタイプ"] or '新馬クラス' in predict_df.iloc[0]['クラス']:
                                continue
                            # selected_cols = Dataset.get_selected_columns(predict_df)
                            # predict_df = predict_df[selected_cols]
                            error = ""
                        elif os.path.isfile(f'{dir_path}/race_table.pickle') and os.path.isfile(f'{dir_path}/peds_table.pickle') and os.path.isfile(f'{dir_path}/training_table.pickle'):
                            df = pd.read_pickle(f'{dir_path}/race_table.pickle')
                            df['日付'] = pd.to_datetime(df['日付']).dt.date
                            TodayData.race_df = df
                            
                            peds_df = pd.read_pickle(f'{dir_path}/peds_table.pickle')
                            TodayData.peds_df = peds_df

                            train_df = pd.read_pickle(f'{dir_path}/training_table.pickle')
                            TodayData.training_df = train_df
                    
                            race_table = RaceTable.preprocess_core(TodayData.race_df, is_pred=True)
                            pedigree_table = PedigreeTable.preprocess_core(TodayData.peds_df)
                            training_table = TrainingTable.preprocess_core(TodayData.training_df)
                    
                            predict_df, error = main.create_race_table(race_id, start_time, race_table, pedigree_table, training_table)
                        else:
                            if os.path.isdir(dir_path) == False:
                                print(f'create is {dir_path} folder')
                                # フォルダ作成
                                os.makedirs(dir_path)
                            # 
                            NetKeibaAccesser.run(TodayData.scrape_race_table, day=main.today, now_race_id=race_id)
                            # 前処理
                            predict_df, error = main.preprocess(race_id, start_time, TodayData.race_df, TodayData.peds_df, TodayData.training_df)

                        # 対象外の場合
                        if isinstance(predict_df, pd.DataFrame) == False:
                            print(f'{error} {race_id}')
                            print()
                            continue
                    
                        predict_df2, error = main.predict(race_id, predict_df, objective)
                        if isinstance(predict_df2, pd.DataFrame) == False:
                            print(f'{error}')
                            print()
                            continue
                        
                        # マージ
                        result_df = pd.concat([result_df, predict_df2])        
                        print()
        except Exception as e:
            print(f'error: {e}')
            Notification.send(f'error: {e}')
        else:
            Notification.send('シミュレーション作成 完了')

    @staticmethod
    def get_race_result(days: list, objective: str) -> None:
        """ レース結果取得関数
            Args:
                days (list): 日付のリスト
                objective (str): 目的関数名
            Returns:
                None
        """
        for date in tqdm(days):
            print(date[0], date[-1])
            for today in date:
                print(today)
                main = Forecaster(today, date, objective)
                main.create_time_table()
                race_ids = main.time_table[main.time_table['start_time'].notna()].index.to_list()
                if 'win5' in race_ids:
                    race_ids.remove('win5')

                NetKeibaAccesser.run(TodayData.scrape_refund_table, now_race_ids=race_ids)
                str_day = datetime.strftime(main.today, '%Y%m%d')
                dir_path = f'./html/predict/{str_day}'
                if os.path.isdir(dir_path) == False:
                    os.makedirs(dir_path)
                TodayData.result_df.to_pickle(f'{dir_path}/result_{str_day}.pickle')

    @staticmethod
    def predict_result_aggregate_core(obj: Forecaster, objective: str, target_value: str, target_name: str) -> None:
        """ 予測結果集計のコア関数
            Args:
                obj (Forecaster): Forecasterオブジェクト
                objective (str): 目的関数名
                target_value (str): ターゲット値
                target_name (str): ターゲット名
            Returns:
                None
        """
        str_day = datetime.strftime(obj.today, '%Y%m%d')
        result_df = pd.DataFrame()
        for race_id in tqdm(obj.time_table[obj.time_table['start_time'].notna()].index):
            if race_id == 'win5':
                continue

            file_path=f'./html/predict/{str_day}/{race_id}/{objective}/{target_value}/{target_name}.pickle'
            if os.path.isfile(file_path) == False:
                print(f'continue {file_path}')
                continue
            temp_df = pd.read_pickle(file_path)
            print(len(result_df), race_id, len(temp_df))
            result_df = pd.concat([result_df, temp_df])
            
        dir_path=f'./html/predict/{str_day}/{objective}/{target_value}'
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)
        result_df.to_pickle(f'{dir_path}/{target_name}.pickle')
    
    @staticmethod
    def predict_result_aggregate(days: list, objective:str) -> None:
        """ 予測結果集計関数
            Args:
                days (list): 日付のリスト
                objective (str): 目的関数名
            Returns:
                None
        """
        for date in tqdm(days):
            print(date[0], date[-1])
            for today in date:
                print(today)
                main = Forecaster(today, date, objective)
                main.create_time_table()

                for target_value in ['rank3']:
                    Simulator.predict_result_aggregate_core(main, objective, target_value, 'predict_df')


