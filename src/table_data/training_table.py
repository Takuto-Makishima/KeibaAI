import os
import re
import time
from datetime import datetime
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from src.utils.notification import Notification
from src.utils.path_manager import PathManager


class TrainingTable:
    """ 調教表 """

    @staticmethod
    def scraping(driver: WebDriver, days: List[datetime] = [], day: Optional[datetime] = None, now_race_id: str = '', now_race_ids: List[str] = []):
        """ 調教表取得
            Args:
                driver (WebDriver): ドライバ
                days (List[datetime]): 取得対象日
                day (Optional[datetime]): 取得対象日
                now_race_id (str): 取得対象レースID
                now_race_ids (List[str]): 取得対象レースID
            Returns:
                None
        """
        race_ids = []
        lst = [datetime.strftime(i, '%Y%m%d') for i in days]
        for file in lst:
            file_path = f'./html/race_ids/{file[:4]}/{file}.bin'
            with open(file_path) as f:
                temp = f.read()

            soup = BeautifulSoup(temp, 'html.parser')
            data = soup.find_all(class_='RaceList_DataList')
            # 取得したリストを解析
            for element in data:
                # レースID
                data_items = element.find_all(class_='RaceList_DataItem')
                for item in data_items:
                    a = item.find('a').get('href')
                    race_ids.append(re.findall(r"\d+", a)[0])

        # ディレクトリ作成
        data_path = f'./html/training_data/{lst[0][:4]}'
        if os.path.isdir(data_path) == False:
            print(f'create is {data_path} folder')
            os.makedirs(data_path)

        # レースIDループ
        for race_id in tqdm(race_ids):
            try:
                if race_id == '':
                    continue
                # ファイルパス
                file_path = f'{data_path}/{race_id}.bin'
                if os.path.isfile(file_path) == True:
                    print(f'exist {race_id}')
                    continue

                url = f'https://race.netkeiba.com/race/oikiri.html?race_id={race_id}'
                for _ in range(3):
                    try:
                        driver.get(url)
                        # データ取得まで待つ
                        element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'race_table_01')))
                        time.sleep(1)
                    except Exception as e:
                        print(f'{race_id} race_table_01 continue')
                        continue
                    else:
                        # 失敗しなかった場合は、ループを抜ける
                        break
                else:
                    msg = f'取得失敗:{url}'
                    raise TimeoutException(msg)

                data = BeautifulSoup(driver.page_source, 'html.parser')
                encoded = str(data).encode('cp932', "ignore")
                with open(file_path, "wb") as f:
                    f.write(encoded)
            #存在しないrace_idを飛ばす
            except IndexError:
                print(f'IndexError {race_id}')
                Notification.send(f'TrainingTable scraping IndexError {race_id}: {e}')
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(f'Exception {race_id} {e}')
                Notification.send(f'TrainingTable scraping Exception {race_id}: {e}')
                continue
            #Jupyterで停止ボタンを押した時の対処    
            except:
                print(f'Exception {race_id}')
                error = True
                break

    @staticmethod
    def create_data(start: int, end: int) -> None:
        """ データ作成
            Args:
                start (int): 開始年
                end (int): 終了年
            Returns:
                None
        """
        dic_header = {}
        for year in range(start, end):
            df = pd.DataFrame()
            path = f'./html/training_data/{year}'
            # ファイルリスト取得
            files = os.listdir(path)
            # ファイルループ
            for file in tqdm(files):
                html = None
                with open(f'{path}/{file}') as f:
                    html = f.read()
                # インスタンス生成
                soup = BeautifulSoup(html, 'html.parser')
                # tr 取得
                trs = soup.find_all('tr')
                # 長さ取得
                length = len(trs)
                # ヘッダ作成
                ths = trs[0].find_all('th')
                if len(dic_header) == 0:
                    for i in range(0, len(ths)):
                        dic_header[i] = ths[i].get_text(strip=True)

                # データ解析
                dic_data = {}
                for i in range(1, length):
                    # td 取得
                    tds = trs[i].find_all('td')
                    for col_index, col_name in dic_header.items():
                        if col_name == '印':
                            continue
                        if col_name == '馬名':
                            name = tds[col_index].find(href=re.compile(r"https://db.netkeiba.com/horse/\d"))
                            ids = re.findall('[0-9]+', str(name))
                            if col_name in dic_data:
                                dic_data[col_name].append(name.get_text(strip=True))
                                dic_data[f'{col_name}_ID'].append(ids[0])
                            else:
                                # 初回
                                dic_data[col_name] = [name.get_text(strip=True)]
                                dic_data[f'{col_name}_ID'] = [ids[0]]
                        elif col_name == '調教タイムラップ表示':
                            # li 取得
                            lis = tds[col_index].find_all('li')
                            if len(lis) != 0:
                                for count in range(0, len(lis)):
                                    name = f'{col_name}_{count}'
                                    if name in dic_data:
                                        dic_data[name].append(lis[count].get_text(strip=True))
                                    else:
                                        dic_data[name] = [lis[count].get_text(strip=True)]
                        elif col_name == '評価':
                            if col_name in dic_data:
                                dic_data[col_name].append(tds[col_index].get_text(strip=True))
                                dic_data[f'{col_name}欄'].append(tds[col_index+1].get_text(strip=True))
                            else:
                                dic_data[col_name] = [tds[col_index].get_text(strip=True)]
                                dic_data[f'{col_name}欄'] = [tds[col_index+1].get_text(strip=True)]
                        else:
                            if col_name in dic_data:
                                dic_data[col_name].append(tds[col_index].get_text(strip=True))
                            else:
                                dic_data[col_name] = [tds[col_index].get_text(strip=True)]

                # データフレーム変換       
                df_add = pd.DataFrame.from_dict(dic_data)
                df_add.index = [file.replace('.bin', '')] * len(df_add)

                if len(df) == 0:
                    df = df_add
                else:
                    df = pd.concat([df,df_add])

            df.to_pickle(PathManager.get_html_training_table(year))

    @staticmethod
    def cnv_training_rap(x) -> float:
        """ 調教ラップ変換
            Args:
                x (str): 変換対象
            Returns:
                str: 変換後
        """
        if x == "":
            return None
        res = re.findall(r"(?<=\()-?\d+(?:\.\d+)?(?=\))", x)
        if len(res) != 0:
            return float(res[0])
        else:
            return None

    @staticmethod
    def cnv_training_time(x) -> float:
        """ 調教タイム変換
            Args:
                x (str): 変換対象
            Returns:
                str: 変換後
        """
        if x == "":
            return None
        temp = x.split('(')
        if len(temp) == 2 and not temp[0] == "":
            return float(temp[0])
        else:
            return None

    @staticmethod
    def cnv_date(x: str) -> datetime:
        """ 日付変換
            Args:
                x (str): 変換対象
            Returns:
                datetime.date: 変換後
        """
        res = re.findall(r'\b\d{4}/\d{2}/\d{2}\b', x)
        if len(res) != 0:
            tdatetime = datetime.strptime(res[0], '%Y/%m/%d')
            return datetime(tdatetime.year, tdatetime.month, tdatetime.day)
        else:
            return None

    @staticmethod
    def is_float(s: str) -> bool:
        """ 浮動小数点判定
            Args:
                s (str): 判定対象
            Returns:
                bool: 判定結果
        """
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def preprocess_core(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """ 前処理
            Args:
                df (pl.DataFrame): 前処理対象
            Returns:
                pl.DataFrame: 前処理後
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        # 1. 列名の変更
        df = df.rename({
            '枠': '枠番', #'日付': '調教日', 
            '馬場': '調教馬場',
            '調教タイムラップ表示_0': '調教タイム5', '調教タイムラップ表示_1': '調教タイム4',
            '調教タイムラップ表示_2': '調教タイム3', '調教タイムラップ表示_3': '調教タイム2',
            '調教タイムラップ表示_4': '調教タイム1'
        })

        # 2. 日付変換と型変換
        df = df.with_columns([
            pl.col("日付").map_elements(lambda x: TrainingTable.cnv_date(x), return_dtype=pl.Datetime).alias("調教日"),
            pl.col("枠番").cast(pl.Int32),
            pl.col("馬番").cast(pl.Int32)
        ])

        # 3. 調教ラップの変換
        for i in range(1, 6):
            df = df.with_columns([
                pl.col(f"調教タイム{i}").map_elements(lambda x: TrainingTable.cnv_training_rap(x), return_dtype=pl.Float64).alias(f"調教ラップ_{i}")
            ])

        # 4. 調教タイムの変換
        for i in range(1, 6):
            df = df.with_columns([
                pl.col(f"調教タイム{i}").map_elements(lambda x: TrainingTable.cnv_training_time(x), return_dtype=pl.Float64).alias(f"調教タイム_{i}")
            ])

        df = df[['race_id', '枠番', '馬番', '馬名', '馬名_ID', '調教日', 'コース', '調教馬場', '乗り役', '調教タイム_5', '調教タイム_4', '調教タイム_3', '調教タイム_2', '調教タイム_1', '調教ラップ_5', '調教ラップ_4', '調教ラップ_3', '調教ラップ_2', '調教ラップ_1', '位置', '脚色', '評価', '評価欄']]
        
        return df

    @staticmethod
    def preprocess(start: int, end: int) -> None:
        """ 前処理
            Args:
                start (int): 開始年
                end (int): 終了年
            Returns:
                None
        """
        for year in tqdm(range(start, end)):
            print(year)
            training_table = pd.read_pickle(PathManager.get_html_training_table(year))
            training_table['race_id'] = training_table.index
            training_table = pl.from_pandas(training_table)
            
            df = TrainingTable.preprocess_core(training_table)

            save_path = PathManager.get_training_table_extra(year, False)
            df.write_parquet(save_path)
