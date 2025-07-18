import os
import re
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import polars as pl

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


if __name__ == '__main__':

    DAYS_LIST = [
        # datetime(2025,1,5).date(), datetime(2025,1,6).date(),
        # datetime(2025,1,11).date(), datetime(2025,1,12).date(), datetime(2025,1,13).date(),
        # datetime(2025,1,18).date(), datetime(2025,1,19).date(),
        # datetime(2025,1,25).date(), datetime(2025,1,26).date(),
        # datetime(2025,2,1).date(), datetime(2025,2,2).date(),
        datetime(2025,2,8).date(), datetime(2025,2,9).date(), datetime(2025,2,10).date(),
        datetime(2025,2,15).date(), datetime(2025,2,16).date(),
        datetime(2025,2,22).date(), datetime(2025,2,23).date(),
    ]
    START_YEAR = 2025
    END_YEAR = 2026
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    Notification.send('学習データ作成 開始')

    try:
        print('race_id スクレイピング')
        NetKeibaAccesser.run(RaceId.scraping, days=DAYS_LIST)
        print('race_id データ作成')
        RaceId.create_data(START_YEAR, END_YEAR)
        print('出馬表 スクレイピング')
        NetKeibaAccesser.run(RaceTable.scraping, days=DAYS_LIST)
        print('出馬表 データ作成')
        RaceTable.create_data(START_YEAR, END_YEAR)
        print('払戻表 データ作成')
        RefundTable.create_data(START_YEAR, END_YEAR)
        print('血統表 スクレイピング')
        NetKeibaAccesser.run(PedigreeTable.scraping, days=DAYS_LIST)
        print('血統表 データ作成')
        PedigreeTable.create_data()
        print('調教 スクレイピング')
        NetKeibaAccesser.run(TrainingTable.scraping, days=DAYS_LIST)
        print('調教 データ作成')
        TrainingTable.create_data(START_YEAR, END_YEAR)
        print('出馬表 前処理')
        RaceTable.preprocess(START_YEAR, END_YEAR)
        print('血統表 前処理')
        PedigreeTable.preprocess()
        print('調教 前処理')
        TrainingTable.preprocess(START_YEAR, END_YEAR)
        print('前処理 マージ')
        RaceTable.merge_preprocess_data(START_YEAR, END_YEAR)
        print('ホースレート データ作成')
        HorseRateData.create_data(END_YEAR)
        print('過去データ作成')
        PastData.create_past_data_extra(START_YEAR, END_YEAR)
        print('統計データ作成')
        StaticticsData.create_statictics_data_extra(START_YEAR, END_YEAR)
        print('ランクデータ作成')
        RankingData.create_rank_data_extra(START_YEAR, END_YEAR)

    except Exception as e:
        Notification.send(f'学習データ: {e}')
    else:
        Notification.send('学習データ作成 完了')

