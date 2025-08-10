import os
import time
import requests
from datetime import datetime
import pandas as pd
import polars as pl
import schedule
from selenium.webdriver import Edge
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from src.utils.path_manager import PathManager
from src.utils.netkeiba_accesser import NetKeibaAccesser
from src.utils.notification import Notification
from src.utils.json_serializer import JsonSerializer
from src.creators.ai_creator import AiCreator
from src.executors.today_data import TodayData
from src.executors.user_info import UserInfo
from src.creators.dataset import Dataset
from src.organize_data.past_data import PastData
from src.organize_data.statictics_data import StaticticsData
from src.table_data.race_table import RaceTable
from src.table_data.pedigree_table import PedigreeTable
from src.table_data.training_table import TrainingTable
from src.organize_data.ranking_data import RankingData
from src.utils.settings import Settings


class Forecaster:
    """ 予測を行うクラス """
    def __init__(self, today, days, objective, user_list=[], exec_dict={}, 
                 horse_rate_df:pl.DataFrame=None, aggregate_df:pl.DataFrame=None, time_table=None):
        self.today = today
        self.days = days
        self.objective = objective
        self.user_list = user_list
        self.exec_dict = exec_dict
        self.target_col = 'pred'
        self.win5_msg = ''
        
        self.is_exclusion_new_horse = True
        self.place_table = { '01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉' }
        
        print('horse_rate_df')
        if isinstance(horse_rate_df, pl.DataFrame) == True:
            self.horse_rate_df = horse_rate_df.clone()
        else:
            self.horse_rate_df = horse_rate_df
            
        print('aggregate_df')
        if isinstance(aggregate_df, pd.DataFrame) == True:
            self.aggregate_df = aggregate_df.clone()
        else:
            self.aggregate_df = aggregate_df
        
        print('time_table')
        self.time_table = time_table
        if isinstance(time_table, pd.DataFrame) == True:
            self.time_table = time_table.copy()
        else:
            self.time_table = time_table

    # データ読込
    def load_data(self):
        # ユーザリスト作成
        if len(self.user_list) == 0:
            print('Create user list')
            # 三井住友
            user = JsonSerializer.read('./html/data/maxi_info.json')
            user_info = UserInfo.from_dict(user)            
            self.user_list.append(user_info)
        else:
            print('Exists user list')
        
        # AIリスト取得
        if len(self.exec_dict) == 0:
            print('Scrape load ai list')
            NetKeibaAccesser.run(TodayData.scrape_create_ai_list_split, days=self.days)
            self.exec_dict = TodayData.exec_dict
        else:
            print('Exists ai list')

        read_path=PathManager.get_horse_rate_extra(False)
        self.horse_rate_df = pl.read_parquet(read_path)
        self.horse_rate_df = self.horse_rate_df.filter(pl.col("日付") < pd.Timestamp(self.days[0]))

        self.aggregate_df = StaticticsData.init(self.days[0].year)
        self.aggregate_df = self.aggregate_df.filter(pl.col("日付") < pd.Timestamp(self.days[0]))
        
        self.places = []
        for key, values in self.exec_dict.items():
            for place in values:
                if place not in self.places:
                    self.places.append(place)
        self.sample_df = pd.DataFrame()
        for year in range(2010, self.today.year+1):
            temp = pd.read_pickle(PathManager.get_dataset_extra(year, True))
            temp = temp[temp['会場'].isin(self.places)]
            self.sample_df = pd.concat([self.sample_df, temp])
        day = pd.to_datetime(self.days[0])
        self.sample_df = self.sample_df[self.sample_df['日付'] < day]

    # タイムテーブル作成
    def create_time_table(self):
        if isinstance(self.time_table, pd.DataFrame) == False:
            # タイムテーブルスクレイピング
            NetKeibaAccesser.run(TodayData.scrape_time_table, day=self.today)
            self.time_table = TodayData.time_table.copy()
        else:
            print('Exists time_table')

    # 初期化
    def initialize(self):
        
        self.load_data()
        
        self.create_time_table()

    # 作成
    def create_race_table(self, race_id: str, start_time: datetime, 
                          race_table: pl.DataFrame, pedigree_table: pl.DataFrame, training_table: pl.DataFrame):
        place, race_type, grade, dist, race_number = race_table.select(
            ['会場', 'レースタイプ', 'クラス', '距離', 'レース番号']
        ).row(0)
        dist = int(dist)
        horse_ids = race_table.get_column('馬名_ID').unique(maintain_order=True)

        # マージ
        print('マージ')
        race_table = race_table.join(pedigree_table,  left_on='馬名_ID',  right_on='馬名_ID', how='left')
        race_table = race_table.join(training_table,  on=['race_id', '馬名', '馬名_ID', '枠番', '馬番'],  how='left')

        if (self.is_exclusion_new_horse == True) and (grade == '新馬クラス'):
            msg = 'Prediction result\r\n'
            msg += f'{self.today.year:04}年{self.today.month:02}月{self.today.day:02}日\r\n'
            msg += f'出走時刻 {start_time}\r\n'
            msg += f'{place} {race_number}R\r\n'
            msg += f'新馬クラス'
            return None, msg
        
        if race_type == '障害':
            msg = 'Prediction result\r\n'
            msg += f'{self.today.year:04}年{self.today.month:02}月{self.today.day:02}日\r\n'
            msg += f'出走時刻 {start_time}\r\n'
            msg += f'{place} {race_number}R\r\n'
            msg += f'障害レース'
            return None, msg
        
        # 過去データ追加
        print('過去データ追加')
        date = race_table.select('日付').item(0, 0)
        race_table = PastData.create_past_data_extra_core(self.today, date, race_table, self.horse_rate_df)

        # レート追加
        print('レート追加')
        horse_ids = race_table.get_column('馬名_ID').to_list()
        day = self.days[0]
        latest_rate_df = (
            self.horse_rate_df
            .filter(
                (pl.col("日付") < day) &
                (pl.col("馬名_ID").is_in(horse_ids))
            )
            .sort(["馬名_ID", "日付"], descending=[False, True])
            .unique(subset=["馬名_ID"], keep="first")
            .select(["馬名_ID", pl.col("出走後馬レート").alias("出走前馬レート")])
        )
        race_table = race_table.join(latest_rate_df, on="馬名_ID", how="left")

        rate_mean = race_table.select(pl.col("出走前馬レート").mean()).item()
        if rate_mean != 0.0:
            race_table = race_table.with_columns(
                ((pl.col("出走前馬レート") - rate_mean) / rate_mean * 100).alias("出走前レース内馬レート")
            )
        else:
            race_table = race_table.with_columns([
                pl.lit(None).alias("出走前レース内馬レート")
            ])

        grouped = (
            race_table.group_by("race_id")
            .agg([
                pl.len().alias("出走前レース内馬レート_カウント"),
                pl.col("出走前レース内馬レート").mean().alias("出走前レース内馬レート_平均値"),
                pl.col("出走前レース内馬レート").median().alias("出走前レース内馬レート_中央値"),
                pl.col("出走前レース内馬レート").sum().alias("出走前レース内馬レート_合計"),
                pl.col("出走前レース内馬レート").std().alias("出走前レース内馬レート_標準偏差"),
                pl.col("出走前レース内馬レート").var().alias("出走前レース内馬レート_分散"),
                pl.col("出走前レース内馬レート").min().alias("出走前レース内馬レート_最小値"),
                pl.col("出走前レース内馬レート").max().alias("出走前レース内馬レート_最大値"),
            ])
        )

        # `race_id` をキーにして `result_df` に統計情報を結合（左結合）
        race_table = race_table.join(grouped, on="race_id", how="left")

        print('統計データ追加')
        race_table = StaticticsData.create_statistics_data_core_extra(self.aggregate_df, self.days[0], race_table)
        
        print('ランキングデータ追加')
        race_table = RankingData.create_rank_data_extra_core(race_table)

        print('学習データ追加')
        race_table = Dataset.create_data_core(race_table, True, False)

        str_day = datetime.strftime(self.today, '%Y%m%d')
        dir_path = f'./html/predict/{str_day}/{race_id}'
        if os.path.isdir(dir_path) == False:
            print(f'create is {dir_path} folder')
            os.makedirs(dir_path)
        file_path = f'{dir_path}/predict_dataset.parquet'
        race_table.write_parquet(file_path)

        file_path = f'{dir_path}/predict_dataset.pickle'
        race_table = race_table.to_pandas()
        race_table.to_pickle(file_path)
        
        return race_table, ''

    # 前処理
    def preprocess(self, race_id, start_time, race_df, peds_df, training_df):       
        # 前処理
        race_table = RaceTable.preprocess_core(race_df, is_pred=True)
        pedigree_table = PedigreeTable.preprocess_core(peds_df)
        training_table = TrainingTable.preprocess_core(training_df)
        
        return self.create_race_table(race_id, start_time, race_table, pedigree_table, training_table)

    # 
    def save_predict_df(self, race_id: str, target_value: str, predict_df: pd.DataFrame, objective: str): #, is_contain_nan_past3, is_contain_nan_past2):
        # 予測結果保存
        str_day = datetime.strftime(self.today, '%Y%m%d')
        dir_path = f'./html/predict/{str_day}/{race_id}/{objective}/{target_value}'
        if os.path.isdir(dir_path) == False:
            print(f'create is {dir_path} folder')
            os.makedirs(dir_path)
        print(f'save {dir_path}/predict_df.pickle')
        predict_df.to_pickle(f'{dir_path}/predict_df.pickle')
        # if is_contain_nan_past3==False:
        #     predict_df.to_pickle(f'{dir_path}/predict_df_past3_all_ok.pickle')
        # if is_contain_nan_past2==False:
        #     predict_df.to_pickle(f'{dir_path}/predict_df_past2_all_ok.pickle')
        pass
        
    # 予測
    def predict(self, race_id: str, src_df: pd.DataFrame, objective: str):
        try:
            df = Dataset.convert_column_type_pd(src_df, True, True)
            place = df['会場'].iloc[0]
            race_type = df['レースタイプ'].iloc[0]
            grade = df['クラス'].iloc[0]
            distance = int(df['距離'].iloc[0])
            race_number = df['レース番号'].iloc[0]
            start_day=self.days[0]
            end_day=self.days[-1]
            dis = Dataset.convert_distance(race_type, place, distance)

            for target_value in ['rank3']:
                path = PathManager.get_model_path(start_day, end_day, objective, target_value, race_type, place, int(dis))
                model = pd.read_pickle(path)
                if 'race_id' in df.columns:
                    df = df.drop(columns=['race_id'])
                predict_df = df.copy()
                predict_df.loc[:, 'pred'] = model.predict(df, num_iteration=model.best_iteration)
                cols=['日付', '会場', 'レースタイプ', '距離', 'クラス', '枠番', '馬番', '馬場']
                predict_df[cols] = src_df[cols]
                predict_df['race_id'] = race_id

                # is_contain_nan_past3 = df['3走前_order'].isna().any()
                # is_contain_nan_past2 = df['2走前_order'].isna().any()
                self.save_predict_df(race_id, target_value, predict_df, objective)# , is_contain_nan_past3, is_contain_nan_past2)

            return predict_df, ''
            
        except Exception as e:
            m_r = set(model.feature_name()) - set(df.columns)
            r_m = set(df.columns) - set(model.feature_name())
            return None, f'{race_id} predict error\n{e}\nm-r = {m_r}\nr-m = {r_m}'

    # メッセージ作成
    def create_message(self, result_df, start_time, is_win5=False):
        # メッセージ作成
        place = result_df['会場'].iloc[0]
        race_num = int(result_df['レース番号'].iloc[0])
        grade = result_df['クラス'].iloc[0]
        race_type = result_df['レースタイプ'].iloc[0]
        distance = int(result_df['距離'].iloc[0])
        race_name = TodayData.race_df['レース名'].iloc[0]

        msg = ''
        target_count = 3
        if is_win5 == False:
            msg = 'Prediction result\r\n'
            msg += f'{self.today.year:04}年{self.today.month:02}月{self.today.day:02}日\r\n'
            msg += f'出走時刻 {start_time}\r\n'
            target_count = 5
            
        msg += f'{place} {race_num}R\r\n'
        msg += f'{race_name}\r\n'
        msg += f'{race_type} {distance} {grade}\r\n'
        
        # 予測テーブル作成失敗時
        if isinstance(result_df, pd.DataFrame) == False:
            print('予測対象外')
            msg += '予測対象外'
            Notification.send(msg)
            return None

        cnt = 1
        msg += 'No. 枠番 馬番 人気 値'
        rows = result_df.head(target_count)
        #for index, frame, horse, pop, odds, pred in zip(rows.index, rows['枠番'], rows['馬番'], rows['人気'], rows['補正オッズ'], rows[self.target_col]):                
        for index, frame, horse, pop, pred in zip(rows.index, rows['枠番'], rows['馬番'], rows['人気'], rows[self.target_col]):                
            msg += '\r\n'
            msg += f'{cnt:02}.  {int(frame):02}  {int(horse):02}  {int(pop):02} {pred:.02f}'
            cnt += 1

        return msg

    # Line通知
    def line_notify(self, message, token='6blTmxD35r6zTeq1uG9h31GEcpO5JuXOluyRWZiaUQw'):
        #line_notify_token = token
        #line_notify_api = 'https://notify-api.line.me/api/notify'
        #payload = {'message': message}
        #headers = {'Authorization': 'Bearer ' + line_notify_token}
        #requests.post(line_notify_api, data=payload, headers=headers)
    
        url = 'https://discord.com/api/webhooks/1309626579579830344/SVyePgww0pAXVD6arRMIqWcTs3GxXxcHolYc4pEsULa0veTa4FqQ16q3S4G5HVqTJhah'
        data = {"content": message}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"エラー：{e}")

    # 通常
    def bet_normal(self, driver, user, frame1, frame2, frame3, number1, number2, number3):
            
        # bet_type
        # normal
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[1]/a
        # box
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[4]/a
        bet_type = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[1]/a')
        bet_type.click()
        time.sleep(1)
        
        # 単勝
        # //*[@id="mark-anchor"]/td[1]/button
        #bracket = driver.find_element(by=By.XPATH, value='//*[@id="mark-anchor"]/td[1]/button')
        # 複勝
        # //*[@id="mark-anchor"]/td[2]/button
        # 3連複
        # //*[@id="mark-anchor"]/td[8]/button
        # 3連単
        # //*[@id="mark-anchor"]/td[9]/button
        # 3連複 検索
        bracket = driver.find_element(by=By.XPATH, value='//*[@id="mark-anchor"]/td[8]/button')
        # クリック
        bracket.click()
        time.sleep(1)
        
        # 左上段 1 ～ 9
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[1]/div[1]/button/span
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[1]/div[9]/button/span
        # 左下段 10 ～ 18
        #//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[2]/div[1]/button/span
        #//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[2]/div[9]/button/span

        # 中上段 1 ～ 9
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[1]/div[1]/button/span
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[1]/div[9]/button/span
        # 中下段 10 ～ 18
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[2]/div[1]/button/span
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[2]/div[9]/button/span

        # 下上段 1 ～ 9
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[3]/div/div[1]/div[1]/button/span
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[3]/div/div[1]/div[9]/button/span
        # 下下段 10 ～ 18
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[3]/div/div[2]/div[1]/button/span
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[3]/div/div[2]/div[9]/button/span
        
        first = None
        if number1 <= 9:
            first = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[1]/div[{}]/button/span'.format(number1))
        else:
            first = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[2]/div[{}]/button/span'.format(number1-9))    
        first.click()
        time.sleep(1)

        second = None
        if number2 <= 9:
            second = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[1]/div[{}]/button/span'.format(number2))
        else:
            second = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[2]/div[{}]/button/span'.format(number2-9))    
        second.click()
        time.sleep(1)

        third = None
        if number3 <= 9:
            third = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[3]/div/div[1]/div[{}]/button/span'.format(number3))
        else:
            third = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[3]/div/div[2]/div[{}]/button/span'.format(number3-9))
        third.click()
        time.sleep(1)

        # 枠連
        #first = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[1]/div/div[1]/div[{}]/button/span'.format(frame1))
        #first.click()
        #time.sleep(1)

        #second = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/div/div[1]/div[{}]/button/span'.format(frame2))
        #second.click()
        #time.sleep(1)

        # 金額 上段 30 20 10 5
        # 30
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[2]/div[2]/div[1]/button[1]/span
        # 5
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[2]/div[2]/div[1]/button[4]/span
        # 金額 下段 4 3 2 1
        # 4
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[2]/div[2]/div[2]/button[1]/span
        # 1
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[2]/div[2]/div[2]/button[4]/span

        # レート
        rate = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[2]/div[2]/div[2]/button[4]/span')
        rate.click()
        time.sleep(1)
        
        # 単位 万
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[1]/button
        # 単位 千
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[2]/button
        # 単位 百
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[3]/button

        # 単位
        money = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[3]/button')
        money.click()
        time.sleep(1)

        # セット
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/select-list/div/div/div[3]/div[4]/button[1]
        set_button = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/select-list/div/div/div[3]/div[4]/button[1]')
        set_button.click()
        time.sleep(1)

    # ボックス
    def bet_box(self, driver, user, frame1, frame2, frame3, frame4, number1, number2, number3, number4):
        # normal
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[1]/a
        # box
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[4]/a
        card_type = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[4]/a')
        card_type.click()
        time.sleep(1)
        
        # 
        # //*[@id="mark-anchor"]/td[5]/button
        bet_type = driver.find_element(by=By.XPATH, value='//*[@id="mark-anchor"]/td[5]/button')
        bet_type.click()
        time.sleep(1)
        
        # 1
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[1]/td[1]/button/span
        # 9
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[1]/td[9]/button/span
        # 10
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[2]/td[1]/button/span
        # 18
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[2]/td[9]/button/span
        first = None
        if number1 <= 9:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[1]/td[{number1}]/button/span')
        else:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[2]/td[{number1-9}]/button/span')
        first.click()
        time.sleep(1)

        first = None
        if number2 <= 9:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[1]/td[{number2}]/button/span')
        else:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[2]/td[{number2-9}]/button/span')
        first.click()
        time.sleep(1)
        
        first = None
        if number3 <= 9:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[1]/td[{number3}]/button/span')
        else:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[2]/td[{number3-9}]/button/span')
        first.click()
        time.sleep(1)
        
        first = None
        if number4 <= 9:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[1]/td[{number4}]/button/span')
        else:
            first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div[2]/table/tbody/tr[2]/td[{number4-9}]/button/span')
        first.click()
        time.sleep(1)
        
        # レート
        # 5
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div[2]/div[1]/button[4]/span
        # 1
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div[2]/div[2]/button[4]/span
        rate = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div[2]/div[2]/button[4]/span')
        rate.click()
        time.sleep(1)
        
        # 単位
        money = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div[2]/div/div[3]/button')
        money.click()
        time.sleep(1)
        
        # セット
        set_button = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/select-list/div/div/div[3]/div[4]/button[1]')
        set_button.click()
        time.sleep(1)

    # フォーメーション
    def bet_formation(self, driver, user, frame1, frame2, frame3, frame4, number1, number2, number3, number4):
        # normal
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[1]/a
        # box
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[4]/a
        # formation
        # //*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[5]/a
        card_type = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/nav/ul/li[5]/a')
        card_type.click()
        time.sleep(1)
        
        # 3連複
        # //*[@id="mark-anchor"]/td[5]/button
        # 3連単
        # //*[@id="mark-anchor"]/td[6]/button
        bet_type = driver.find_element(by=By.XPATH, value='//*[@id="mark-anchor"]/td[6]/button')
        bet_type.click()
        time.sleep(1)
        
        # 1着目 1
        #//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[1]/table/tbody/tr/td[1]/div[1]/button
        # 1着目 10
        #//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[1]/table/tbody/tr/td[1]/div[10]/button
        first = None
        first = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[1]/table/tbody/tr/td[1]/div[{number1}]/button')
        first.click()
        time.sleep(1)

        # 2着目 1
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[2]/table/tbody/tr/td[1]/div[1]/button
        # 2着目 10
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[2]/table/tbody/tr/td[1]/div[10]/button

        second = None
        second = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[2]/table/tbody/tr/td[1]/div[{number2}]/button')
        second.click()
        time.sleep(1)
        
        second = None
        second = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[2]/table/tbody/tr/td[1]/div[{number3}]/button')
        second.click()
        time.sleep(1)

        # 3着目 1
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[3]/table/tbody/tr/td[1]/div[1]/button
        # 3着目 10
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[3]/table/tbody/tr/td[1]/div[10]/button
        third = None
        third = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[3]/table/tbody/tr/td[1]/div[{number2}]/button')
        third.click()
        time.sleep(1)
        third = None
        third = driver.find_element(by=By.XPATH, value=f'//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[1]/div/div[3]/table/tbody/tr/td[1]/div[{number3}]/button')
        third.click()
        time.sleep(1)
        
        # レート
        # 5
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div[2]/div[1]/button[4]
        # 1
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div[2]/div[2]/button[4]
        rate = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div[2]/div[2]/button[4]')
        rate.click()
        time.sleep(1)
        
        # 単位
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div[2]/div/div[3]/button
        money = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div[2]/div/div[3]/button')
        money.click()
        time.sleep(1)
        
        # セット
        # //*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/select-list/div/div/div[3]/div[4]/button[1]
        set_button = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/ui-view/ui-view/main/div/div[2]/select-list/div/div/div[3]/div[4]/button[1]')
        set_button.click()
        time.sleep(1)

    # 自動購入
    def automatic_purchase(self, race_id, frame1, frame2, frame3, frame4, number1, number2, number3, number4):
        for user in self.user_list:
            url = 'https://www.ipat.jra.go.jp/'
            
            # クロームのオプション生成
            #options = ChromeOptions()
            # ブラウザ非表示設定
            #options.add_argument('--headless')
            # Chrome生成
            #driver = Chrome(ChromeDriverManager().install(), options=options)

            options = EdgeOptions()
            options.add_argument("--disable-javascript")
            options.add_argument("-–disable-extensions")
            options.add_argument("--disable-background-networking")
            options.add_experimental_option('extensionLoadTimeout', 10000)
            #options.add_argument('--headless')
            
            driver = Edge(options=options)

            for i in range(3):
                #タイムアウトのハンドリング
                try:
                    driver.get(url)
                    time.sleep(1)
                except TimeoutException:
                    continue
                else:
                    # 失敗しなかった場合は、ループを抜ける
                    break
            else:
                msg = f'取得失敗(最大実行回数オーバー):{url}'
                raise Exception(msg)
            
            # INET-ID 検索
            i_net_id = driver.find_element(by=By.XPATH, value='//*[@id="top"]/div[3]/div/table/tbody/tr/td[2]/div/div/form/table[1]/tbody/tr/td[2]/span/input')
            time.sleep(1)
            # INET-ID 入力
            i_net_id.send_keys(user.i_net_id)
            time.sleep(1)
            
            # ログインボタン検索
            login = driver.find_element(by=By.XPATH, value='//*[@id="top"]/div[3]/div/table/tbody/tr/td[2]/div/div/form/table[1]/tbody/tr/td[3]/p')
            # ログインボタンクリック
            login.click()
            time.sleep(1)
            
            # 加入番号 検索
            entry_num = driver.find_element(by=By.XPATH, value='//*[@id="main_area"]/div/div[1]/table/tbody/tr[1]/td[2]/span/input')
            time.sleep(1)
            # 加入番号 入力
            entry_num.send_keys(user.entry_num)
            time.sleep(1)
            
            # 暗証番号 検索
            password = driver.find_element(by=By.XPATH, value='//*[@id="main_area"]/div/div[1]/table/tbody/tr[2]/td[2]/span/input')
            time.sleep(1)
            # 暗証番号 入力
            password.send_keys(user.password)
            time.sleep(1)
            
            # P-ARS番号 検索
            p_ars_num = driver.find_element(by=By.XPATH, value='//*[@id="main_area"]/div/div[1]/table/tbody/tr[3]/td[2]/span/input')
            time.sleep(1)
            # P-ARS番号 入力
            p_ars_num.send_keys(user.p_ars_num)
            time.sleep(1)
            
            # ネット投票メニューへボタン 検索
            net_menu = driver.find_element(by=By.XPATH, value='//*[@id="main_area"]/div/div[1]/table/tbody/tr[1]/td[3]/p/a')
            time.sleep(1)
            # ネット投票メニューへボタン クリック
            net_menu.click()
            time.sleep(1)
            
            #カレントページのURLを取得
            cur_url = driver.current_url

            # お知らせページの有無
            if 'announce' in str(cur_url):
                #カレントページのURLを表示
                print(cur_url)
                # お知らせの OK ボタン検索
                announce_ok = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div[2]/div[2]/button')
                time.sleep(1)
                # マークカード投票ボタン クリック
                announce_ok.click()
                time.sleep(1)
            
            # マークカード投票ボタン 検索
            mark_card = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/main/div[2]/div[1]/div[3]/button')
            time.sleep(1)
            # マークカード投票ボタン クリック
            mark_card.click()
            time.sleep(1)
            
            # 会場のボタン 検索
            buttons = driver.find_elements(by=By.CLASS_NAME, value='place-name')
            # 会場名作成
            place_name = self.place_table[race_id[4:6]]
            # 曜日取得
            weekday = TodayData.get_weekday(self.today.weekday())
            # ボタンのテキスト作成
            button_text = '{0}（{1}）'.format(place_name, weekday)
            selected = None
            
            print('target = ' + button_text)
            
            # 対象のボタン検索
            for item in buttons:
                if item.text == button_text:
                    selected = item
                    break
            if selected == None:
                pass
            
            # 対象のボタンクリック
            selected.click()
            time.sleep(1)
            
            race_num = int(race_id[10:12])
            race_text = f'{race_num}R'
            races = driver.find_elements(by=By.CLASS_NAME, value='race-no')
            race_button = None
            print(f'target = {race_text}')
            
            for item in races:
                if race_text in item.text:
                    race_button = item
                    print(item.text)
                    break
            
            print('selected = ' + race_button.text)
            race_button.click()
            time.sleep(1)

            self.bet_formation(driver, user, int(frame1), int(frame2), int(frame3), int(frame4), int(number1), int(number2), int(number3), int(number4))

            # 購入予定リスト
            buy_list = driver.find_element(by=By.XPATH, value='//*[@id="ipat-navbar"]/div/ng-transclude/div/ul/li/button')
            buy_list.click()
            time.sleep(1)
            
            # 金額
            buy_rate = driver.find_element(by=By.XPATH, value='//*[@id="bet-list-top"]/div[5]/table/tbody/tr[1]/td[5]/div/input')
            buy_rate.clear()
            buy_rate.send_keys(user.rate)
            time.sleep(1)
            
            # 合計金額入力
            # //*[@id="bet-list-top"]/div[5]/table/tbody/tr[4]/td/input
            buy_all = driver.find_element(by=By.XPATH, value='//*[@id="bet-list-top"]/div[5]/table/tbody/tr[4]/td/input')
            sum_rate = user.rate * 100 * 2
            buy_all.send_keys(sum_rate)
            time.sleep(1)            

            # 購入する
            # //*[@id="bet-list-top"]/div[5]/table/tbody/tr[5]/td/button
            push = driver.find_element(by=By.XPATH, value='//*[@id="bet-list-top"]/div[5]/table/tbody/tr[5]/td/button')
            push.click()
            time.sleep(3)
            
            # OK
            # /html/body/error-window/div/div/div[3]/button[1]
            ok = driver.find_element(by=By.XPATH, value='/html/body/error-window/div/div/div[3]/button[1]')
            ok.click()
            time.sleep(1)
            
            # ホーム
            # //*[@id="ipat-navbar"]/div/div[1]/a
            home = driver.find_element(by=By.XPATH, value='//*[@id="ipat-navbar"]/div/div[1]/a')
            home.click()
            time.sleep(1)

            # 購入上限
            #balance = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/main/div[1]/div[1]/div/table/tbody/tr[1]/td[2]')
            #user.balance = balance.text
            #msg = f'info\r\n'
            #msg += f'{place_name} {race_text}\r\n'
            #msg += f'購入オッズ:{odds_text}\r\n'
            #msg += f'残金:{balance.text}'
            #Notification.send(msg)
            
            # ログアウト 検索
            #log_out = driver.find_element(by=By.XPATH, value='//*[@id="main"]/ui-view/div/div[2]/a')
            # ログアウト クリック
            #log_out.click()
            #time.sleep(3)
            
            # ログアウトのOK 検索
            #log_out_ok = driver.find_element(by=By.XPATH, value='/html/body/error-window/div/div/div[3]/button[1]')
            # ログアウトOK クリック
            #log_out_ok.click()
            time.sleep(3)
            
            driver.close()
            driver.quit()

    # 実行レースID取得
    def get_execute_race_id(self):
        # テーブルコピー
        tbl = self.time_table.copy()
        # 現時刻保存
        self.start_job_time = datetime.now()
        print(f'start job time = {self.start_job_time}')
        # レースID
        race_ids = tbl.index.unique(maintain_order=True)

        start_time = None
        target_race_id = ''

        # 予測対象の検出
        for race_id in race_ids:
            if tbl.loc[race_id,'is_executed'] == True:
                continue

            # 実行時間取得
            exec_time = tbl.loc[race_id,'execute_time']
            # 出走時間取得
            start_time = tbl.loc[race_id,'start_time']
            # 60秒の補正
            exec_time_low = exec_time - datetime.timedelta(seconds=60)
            exec_time_high = exec_time + datetime.timedelta(seconds=60)

            # 時間抽出
            low_time = exec_time_low.time()
            now_time = self.start_job_time.time()
            high_time = exec_time_high.time()
            
            target_race_id = race_id
            print(f'race_id = {race_id}')
            print(f'{low_time} <= {now_time} <= {high_time}')

            # 時間が範囲内
            if low_time <= now_time <= high_time:
                tbl.loc[race_id,'is_executed'] = True
                print('実行')
                break
            else:
                tbl.loc[race_id,'is_executed'] = True
                print('実行済み')
                
        # 変更したテーブルの保存
        self.time_table = tbl
        print('is_executed count = {}'.format(len(self.time_table[self.time_table['is_executed'] == True])))
        
        return target_race_id, start_time

    # win5実行
    def execute_win5(self, objective):
        msg = 'win5 実行\r\n'
        for race_id in self.time_table[self.time_table['win5']==True].index:
            start_time = self.time_table.loc[race_id,'start_time']
            print(race_id, start_time)
            try:
                NetKeibaAccesser.run(TodayData.scrape_race_table, day=self.today, now_race_id=race_id)
            except Exception as e:
                Notification.send(f'win5 {race_id} scraping error')
                return None
                
            result_df, error = self.preprocess(race_id, start_time)
            if isinstance(result_df, pd.DataFrame) == False:
                message = f'win5 error {error}'
                print(message)
                Notification.send(message)
                return None

            # 予測
            result_df, error = self.predict(race_id, result_df, start_time, objective)
            if isinstance(result_df, pd.DataFrame) == False:
                message = f'win5 error {error}'
                print(message)
                Notification.send(message)
                return None
            # ソート
            result_df = result_df.sort_values(self.target_col, ascending=False).head(5)

            # メッセージ作成
            msg += self.create_message(result_df, start_time, True)
            msg += '\r\n'
            msg += '\r\n'

            self.win5_msg = msg

        # メッセージ送信
        Notification.send(msg)

    # レース実行
    def execute_race(self, race_id, start_time, objective):
        # データ取得
        try:
            NetKeibaAccesser.run(TodayData.scrape_race_table, day=self.today, now_race_id=race_id)
        except Exception as e:
            Notification.send(f'{race_id} scraping error')
            return None

        if (isinstance(TodayData.race_df, pd.DataFrame) == False) or (isinstance(TodayData.peds_df, pd.DataFrame) == False) or (isinstance(TodayData.training_df, pd.DataFrame) == False):
            Notification.send(f'{race_id} DataFrame is none')
            return None
        
        # 前処理
        result_df, error = self.preprocess(race_id, start_time, TodayData.race_df, TodayData.peds_df, TodayData.training_df)
        if isinstance(result_df, pd.DataFrame) == False:
            message = f'{error}'
            print(message)
            Notification.send(message)
            return None
        
        # 予測
        result_df, error = self.predict(race_id, result_df, objective)
        if isinstance(result_df, pd.DataFrame) == False:
            message = f'{error}'
            print(message)
            Notification.send(message)
            return None
        
        # メッセージ作成
        result_df[['人気', 'オッズ']] = TodayData.race_df[['人気', '単勝']]
        top_data = 4
        result_df = result_df.sort_values(self.target_col, ascending=False).head(5)
        msg = self.create_message(result_df, start_time)
        
        # 自動購入
        #self.automatic_purchase(race_id, 
        #                        result_df['枠番'].iloc[0], result_df['枠番'].iloc[1], result_df['枠番'].iloc[2], result_df['枠番'].iloc[3],
        #                        result_df['馬番'].iloc[0], result_df['馬番'].iloc[1], result_df['馬番'].iloc[2], result_df['馬番'].iloc[3])

        # メッセージ送信
        Notification.send(msg)
        print(msg)

    # 実行
    def job(self):
        # 実行情報取得
        race_id, start_time = self.get_execute_race_id()
        
        if race_id == '':
            message = f'予測対象が検出できません: {race_id}'
            print(message)
            Notification.send(message)
            return None
        elif race_id == 'win5':
            #self.execute_win5()
            pass
        else:
            self.execute_race(race_id, start_time, self.objective)

        # 終了条件
        true_cnt = self.time_table['is_executed'].sum()
        tbl_len = len(self.time_table)
        if true_cnt == tbl_len:
            self.is_loop = False
            print('実行フラグ OFF')

    # スケジュール作成
    def create_schedule(self):
        print('スケジュール作成 開始')
        # nan削除
        self.time_table = self.time_table[self.time_table['start_time'].notna()]
        # スケジュール登録
        for idx, exec_time in zip(self.time_table.index, self.time_table['execute_time']):
            job_time = f'{exec_time.hour:02}:{exec_time.minute:02}'
            schedule.every().day.at(job_time).do(self.job)

        print('スケジュール作成 終了')

    # 実行
    def run(self):
        # 初期化
        self.initialize()
        
        schedule.clear()
                
        # スケジュール作成
        self.create_schedule()

        # ループフラグ ON
        self.is_loop = True
        
        print('メインループ 開始')
        # メインループ
        while self.is_loop:
            # 当該時間にタスクがあれば実行
            schedule.run_pending()
            # 1秒スリープ
            time.sleep(1)
            
        print('メインループ 終了')
