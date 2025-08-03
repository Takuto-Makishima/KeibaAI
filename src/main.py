import os
from datetime import datetime
import pandas as pd
import polars as pl
from tqdm import tqdm
import itertools

from src.utils.netkeiba_accesser import NetKeibaAccesser
from src.utils.path_manager import PathManager
from src.utils.notification import Notification
from src.table_data.race_id import RaceId
from src.table_data.race_table import RaceTable
from src.table_data.pedigree_table import PedigreeTable
from src.table_data.training_table import TrainingTable
from src.table_data.refund_table import RefundTable
from src.organize_data.horse_rate_data import HorseRateData
from src.organize_data.past_data import PastData
from src.organize_data.statictics_data import StaticticsData
from src.organize_data.ranking_data import RankingData
from src.creators.dataset import Dataset
from src.creators.ai_creator import AiCreator
from src.executors.today_data import TodayData
from src.executors.forecaster import Forecaster
from src.executors.simulator import Simulator


def create_dataset(days: list, start_year: int, end_year: int):
    """
    データセット作成関数
    """
    Notification.send('学習データ作成 開始')
    try:
        print('race_id スクレイピング')
        NetKeibaAccesser.run(RaceId.scraping, days=days)
        print('race_id データ作成')
        RaceId.create_data(start_year, end_year)
        print('出馬表 スクレイピング')
        NetKeibaAccesser.run(RaceTable.scraping, days=days)
        print('出馬表 データ作成')
        RaceTable.create_data(start_year, end_year)
        print('払戻表 データ作成')
        RefundTable.create_data(start_year, end_year)
        print('血統表 スクレイピング')
        NetKeibaAccesser.run(PedigreeTable.scraping, days=days)
        print('血統表 データ作成')
        PedigreeTable.create_data()
        print('調教 スクレイピング')
        NetKeibaAccesser.run(TrainingTable.scraping, days=days)
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


def create_rank_model(days: list, start_year: int, end_year: int, objective: str):
    """
    モデル作成関数
    """
    Notification.send('モデル作成 開始')
    try:
        print('モデル作成')
        AiCreator.execute_rank_optuna(start_year, end_year, True, days, True)
    except Exception as e:
        print(f'error: {e}')
        Notification.send(f'error: {e}')
    else:
        Notification.send('モデル作成 完了')


def create_prediction_data(days: list, start_year: int, end_year: int, is_past: bool, objective: str):
    """
    シミュレーション作成関数
    """
    Notification.send('シミュレーション作成 開始')
    target_df = None
    try:
        print('シミュレーション作成')
        for date in tqdm(days):
            print(date[0], date[-1])
            for today in date:
                print(f'日付: {today}')
                main = Forecaster(today, date)
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


def get_race_result(days: list):
    for date in tqdm(days):
        print(date[0], date[-1])
        for today in date:
            print(today)
            main = Forecaster(today, date)
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


def predict_result_aggregate_core(obj: Forecaster, objective, target_value, target_name):
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
    

def predict_result_aggregate(days: list, objective:str):
    for date in tqdm(days):
        print(date[0], date[-1])
        for today in date:
            print(today)
            main = Forecaster(today, date)
            main.create_time_table()

            for target_value in ['rank3']:
                predict_result_aggregate_core(main, objective, target_value, 'predict_df')


def read_predict_result(days: list, objective, target_value, target_name='predict_df'):
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


def read_refund_data(days: list):
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

if __name__ == '__main__':

    START_YEAR = 2025
    END_YEAR = 2026
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    DAYS_LIST = [
        # datetime(2025,1,5).date(), datetime(2025,1,6).date(),
        # datetime(2025,1,11).date(), datetime(2025,1,12).date(), datetime(2025,1,13).date(),
        # datetime(2025,1,18).date(), datetime(2025,1,19).date(),
        # datetime(2025,1,25).date(), datetime(2025,1,26).date(),
        # datetime(2025,2,1).date(), datetime(2025,2,2).date(),
        # datetime(2025,2,8).date(), datetime(2025,2,9).date(), datetime(2025,2,10).date(),
        # datetime(2025,2,15).date(), datetime(2025,2,16).date(),
        # datetime(2025,2,22).date(), datetime(2025,2,23).date(),
        # datetime(2025,3,1).date(), datetime(2025,3,2).date(),
        # datetime(2025,3,8).date(), datetime(2025,3,9).date(),
        # datetime(2025,3,15).date(), datetime(2025,3,16).date(),
        # datetime(2025,3,22).date(), datetime(2025,3,23).date(),
        # datetime(2025,3,29).date(), datetime(2025,3,30).date(),
        # datetime(2025,4,5).date(), datetime(2025,4,6).date(),
        # datetime(2025,4,12).date(), datetime(2025,4,13).date(),
        # datetime(2025,4,19).date(), datetime(2025,4,20).date(),
        # datetime(2025,4,26).date(), datetime(2025,4,27).date(),
        # datetime(2025,5,3).date(), datetime(2025,5,4).date(),
        # datetime(2025,5,10).date(), datetime(2025,5,11).date(),
        # datetime(2025,5,17).date(), datetime(2025,5,18).date(),
        # datetime(2025,5,24).date(), datetime(2025,5,25).date(),
        # datetime(2025,5,31).date(), datetime(2025,6,1).date(),
        # datetime(2025,6,7).date(), datetime(2025,6,8).date(),
        # datetime(2025,6,14).date(), datetime(2025,6,15).date(),
        # datetime(2025,6,21).date(), datetime(2025,6,22).date(),
        # datetime(2025,6,28).date(), datetime(2025,6,29).date(),
        # datetime(2025,7,5).date(), datetime(2025,7,6).date(),
        # datetime(2025,7,12).date(), datetime(2025,7,13).date(),
        datetime(2025,7,19).date(), datetime(2025,7,20).date(),
    ]
    # create_dataset(DAYS_LIST, START_YEAR, END_YEAR)

    CREATE_LIST = [
        # [datetime(2025,1,5).date(), datetime(2025,1,6).date()],
        # [datetime(2025,1,11).date(), datetime(2025,1,12).date(), datetime(2025,1,13).date()],
        # [datetime(2025,1,18).date(), datetime(2025,1,19).date()],
        # [datetime(2025,1,25).date(), datetime(2025,1,26).date()],
        # [datetime(2025,2,1).date(), datetime(2025,2,2).date()],
        # [datetime(2025,2,8).date(), datetime(2025,2,9).date(), datetime(2025,2,10).date()],
        # [datetime(2025,2,15).date(), datetime(2025,2,16).date()],
        # [datetime(2025,2,22).date(), datetime(2025,2,23).date()],
        # [datetime(2025,3,1).date(), datetime(2025,3,2).date()],
        # [datetime(2025,3,8).date(), datetime(2025,3,9).date()],
        # [datetime(2025,3,15).date(), datetime(2025,3,16).date()],
        # [datetime(2025,3,22).date(), datetime(2025,3,23).date()],
        # [datetime(2025,3,29).date(), datetime(2025,3,30).date()],
        # [datetime(2025,4,5).date(), datetime(2025,4,6).date()],
        # [datetime(2025,4,12).date(), datetime(2025,4,13).date()],
        # [datetime(2025,4,19).date(), datetime(2025,4,20).date()],
        # [datetime(2025,4,26).date(), datetime(2025,4,27).date()],
        # [datetime(2025,5,3).date(), datetime(2025,5,4).date()],
        # [datetime(2025,5,10).date(), datetime(2025,5,11).date()],
        # [datetime(2025,5,17).date(), datetime(2025,5,18).date()],
        # [datetime(2025,5,24).date(), datetime(2025,5,25).date()],
        # [datetime(2025,5,31).date(), datetime(2025,6,1).date()],
        # [datetime(2025,6,7).date(), datetime(2025,6,8).date()],
        # [datetime(2025,6,14).date(), datetime(2025,6,15).date()],
        # [datetime(2025,6,21).date(), datetime(2025,6,22).date()],
        # [datetime(2025,6,28).date(), datetime(2025,6,29).date()],
        # [datetime(2025,7,5).date(), datetime(2025,7,6).date()],
        # [datetime(2025,7,12).date(), datetime(2025,7,13).date()],
        [datetime(2025,7,19).date(), datetime(2025,7,20).date()],
    ]
    # create_rank_model(CREATE_LIST, 2010, END_YEAR, 'rank_xendcg') # lambdarank

    # 予測データ作成
    create_prediction_data(CREATE_LIST, START_YEAR, END_YEAR, True, 'rank_xendcg')

    # レース結果取得
    # ↓は過去データがない場合に実行する処理に変更する
    get_race_result(CREATE_LIST)

    # 予測結果集計
    predict_result_aggregate(CREATE_LIST, 'rank_xendcg')

    # 表示データ作成
    pred_results = {}
    for target_value in ['rank3']:
        pred_results[target_value] = read_predict_result(CREATE_LIST, 'rank_xendcg', target_value)
    refund_table_df = read_refund_data(CREATE_LIST)

    tbl_preds = {}
    count = 5
    exists_past_3rd = False
    exists_past_2nd = False
    threshold = 0.0
    is_today = True
    is_expected = False
    is_sort = True
    for target_value in tqdm(['rank3']):
        tbl_pred = Simulator.create_predict_table(pred_results[target_value], count, exists_past_3rd, exists_past_2nd, threshold, is_today, is_expected, 'pred', is_sort)
        tbl_pred = Simulator.add_place_result(tbl_pred, refund_table_df.place)
        tbl_pred = Simulator.add_win_result(tbl_pred, refund_table_df.win)
        tbl_pred = Simulator.add_wide_result(tbl_pred, refund_table_df.wide)
        tbl_pred = Simulator.add_bracket_quinella_result(tbl_pred, refund_table_df.bracket_quinella)
        tbl_pred = Simulator.add_quinella_result(tbl_pred, refund_table_df.quinella)
        tbl_pred = Simulator.add_exacta_result(tbl_pred, refund_table_df.exacta)
        tbl_pred = Simulator.add_trio_result(tbl_pred, refund_table_df.trio)
        tbl_pred = Simulator.add_tierce_result(tbl_pred, refund_table_df.tierce)
        tbl_preds[target_value] = tbl_pred

    df = Simulator.calculation_data(tbl_preds['rank3'])
    print(df)

    box_disp_df = {}
    for target_value in tqdm(['rank3']):    
        disp_df = pd.DataFrame()
        tbl_pred = tbl_preds[target_value]
        # 複勝
        target = 'refund_place_box'
        targets = ['p_h_1', 'p_h_2']
        combo = 1
        pattern = sum(1 for ignore in itertools.combinations(targets, combo))
        print('複勝', pattern)
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table_df.place, targets, combo)
        tbl_pred['refund_place_box'] = tbl['refund_place_box']
        place_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, place_df], axis=1)
        tbl_pred['refund_place_box'] = tbl_pred['refund_place_box'] - (pattern * 100)
        
        # 単勝
        target = 'refund_win_box'
        targets = ['p_h_1', 'p_h_2']
        combo = 1
        pattern = sum(1 for ignore in itertools.combinations(targets, combo))
        print('単勝', pattern)
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table_df.win, targets, combo)
        tbl_pred['refund_win_box'] = tbl['refund_win_box']
        win_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, win_df], axis=1)
        tbl_pred['refund_win_box'] = tbl_pred['refund_win_box'] - (pattern * 100)
        
        # ワイド
        target = 'refund_wide_box'
        targets = ['p_h_1', 'p_h_2', 'p_h_3']
        combo = 2 
        pattern = sum(1 for ignore in itertools.combinations(targets, combo))
        print('ワイド', pattern)
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table_df.wide, targets, combo)
        tbl_pred['refund_wide_box'] = tbl['refund_wide_box']
        wide_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, wide_df], axis=1)
        tbl_pred['refund_wide_box'] = tbl_pred['refund_wide_box'] - (pattern * 100)
        
        # 枠連
        target = 'refund_bracket_quinella_box'
        targets = ['p_f_1', 'p_f_2', 'p_f_3']
        combo = 2 
        pattern = sum(1 for ignore in itertools.combinations(targets, combo))
        print('枠連', pattern)
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table_df.bracket_quinella, targets, combo)
        tbl_pred['refund_bracket_quinella_box'] = tbl['refund_bracket_quinella_box']
        bracket_quinella_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, bracket_quinella_df], axis=1)
        tbl_pred['refund_bracket_quinella_box'] = tbl_pred['refund_bracket_quinella_box'] - (pattern * 100)
        
        # 馬連
        target = 'refund_quinella_box'
        targets = ['p_h_1', 'p_h_2', 'p_h_3']
        combo = 2 
        pattern = sum(1 for ignore in itertools.combinations(targets, combo))
        print('馬連', pattern)
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table_df.quinella, targets, combo)
        tbl_pred['refund_quinella_box'] = tbl['refund_quinella_box']
        quinella_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, quinella_df], axis=1)
        tbl_pred['refund_quinella_box'] = tbl_pred['refund_quinella_box'] - (pattern * 100)
        
        # 馬単
        target = 'refund_exacta_box'
        targets = ['p_h_1', 'p_h_2']
        combo = 2
        pattern = sum(1 for ignore in itertools.permutations(targets, combo))
        print('馬単', pattern)
        tbl = Simulator.add_perm_box(tbl_pred, target, refund_table_df.exacta, targets, combo)
        tbl_pred['refund_exacta_box'] = tbl['refund_exacta_box']
        exacta_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, exacta_df], axis=1)
        tbl_pred['refund_exacta_box'] = tbl_pred['refund_exacta_box'] - (pattern * 100)
        
        # 3連複
        target = 'refund_trio_box'
        targets = ['p_h_1', 'p_h_2', 'p_h_3', 'p_h_4']
        combo = 3
        pattern = sum(1 for ignore in itertools.combinations(targets, combo))
        print('3連複', pattern)
        tbl = Simulator.add_combo_box(tbl_pred, target, refund_table_df.trio, targets, combo)
        tbl_pred['refund_trio_box'] = tbl['refund_trio_box']
        trio_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, trio_df], axis=1)
        tbl_pred['refund_trio_box'] = tbl_pred['refund_trio_box'] - (pattern * 100)
        
        # 3連単
        target = 'refund_tierce_box'
        targets = ['p_h_1', 'p_h_2', 'p_h_3']
        combo = 3
        pattern = sum(1 for ignore in itertools.permutations(targets, combo))
        print('3連単', pattern)
        tbl = Simulator.add_perm_box(tbl_pred, target, refund_table_df.tierce, targets, combo)
        tbl_pred['refund_tierce_box'] = tbl['refund_tierce_box']
        tierce_df = Simulator.calcuration(tbl, target, pattern)
        disp_df = pd.concat([disp_df, tierce_df], axis=1)
        tbl_pred['refund_tierce_box'] = tbl_pred['refund_tierce_box'] - (pattern * 100)

        box_disp_df[target_value] = disp_df
    print(box_disp_df['rank3'])