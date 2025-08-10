import os
import re
import time
from typing import List, Optional, Union
from tqdm import tqdm
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from src.utils.notification import Notification
from src.utils.path_manager import PathManager


class PedigreeTable:
    """ 血統表 """

    @staticmethod
    def scraping(driver: WebDriver, days: List[datetime] = [], day: Optional[datetime] = None, now_race_id: str = '', now_race_ids: List[str] = []):
        """ 血統表取得 
            Args:
                driver (WebDriver): ドライバ
                days (List[datetime]): 取得対象日
                day (Optional[datetime]): 取得対象日
                now_race_id (str): 取得対象レースID
                now_race_ids (List[str]): 取得対象レースID
            Returns:
                None
        """
        year = str(days[0].year)
        race_table = pl.read_parquet(PathManager.get_html_race_table(year, False))
        horse_ids = race_table['馬名_ID'].unique(maintain_order=True).to_list()

        # ホースIDループ
        for horse_id in tqdm(horse_ids):
            try:
                # 血統表
                pedigree_file_path = f'./html/pedigree_data/{horse_id}.bin'
                if os.path.isfile(pedigree_file_path) == True:
                    continue

                # 接続先
                url = f'https://db.netkeiba.com/horse/ped/{horse_id}/'
                for _ in range(3):
                    try:
                        driver.get(url)
                        # データ取得まで待つ
                        element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'blood_table')))
                        time.sleep(1)
                    except Exception as e:
                        print(f'pedigree table {horse_id} continue')
                        continue
                    else:
                        # 失敗しなかった場合は、ループを抜ける
                        break
                else:
                    msg = f'取得失敗:{url}'
                    raise TimeoutException(msg)

                data = BeautifulSoup(driver.page_source, 'html.parser')
                encoded = str(data).encode('cp932', "ignore")
                with open(pedigree_file_path, "wb") as f:
                    f.write(encoded)

            #存在しないrace_idを飛ばす
            except IndexError:
                print(f'IndexError {horse_id}')
                Notification.send(f'PedigreeTable scraping IndexError {horse_id}: {e}')
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(f'Exception {horse_id} {e}')
                Notification.send(f'PedigreeTable scraping Exception {horse_id}: {e}')
                continue
            #Jupyterで停止ボタンを押した時の対処    
            except:
                print(f'Exception {horse_id}')
                break

    @staticmethod
    def create_data():
        """ 血統表データ作成 """
        # ディレクトリ
        path = './html/pedigree_data/'
        # ファイルリスト取得
        files = os.listdir(path)
        # pickle データ作成
        for file in tqdm(files):
            try:
                if '.bin' not in file:
                    print(f'Not target {file}')
                    continue
                # index設定
                index = file.replace('.bin', '')
                if os.path.isfile(f'./html/pedigree_table/{index}.pickle') == True:
                    continue

                print(f'not exists {file}')

                # ファイル読込
                html = None
                with open(f'{path}/{file}') as f:
                    html = f.read()
                # インスタンス生成
                soup = BeautifulSoup(html, 'html.parser')
                # テーブル取得
                table = soup.find(class_='blood_table')
                # td取得
                tds = table.find_all('td')
                if len(tds) != 62:
                    print(f'len != 62, {file}')

                cnt = 0
                dic_ped = {}
                for td in tds:
                    name = td.find(href=re.compile(r"/horse/\d"))
                    ids = re.findall('[0-9a-zA-Z]+', str(name))
                    if cnt == 0:
                        types = td.get_text(',', strip=True).split(',')[-1]
                        if '系' in types:
                            dic_ped['peds_type'] = [types]
                        else:
                            dic_ped['peds_type'] = [np.nan]
                            print(f'{file} is none peds_type')

                    if type(name) != type(None):
                        dic_ped[f'peds_{cnt:02}'] = [name.get_text(strip=True)]
                    else:
                        dic_ped[f'peds_{cnt:02}'] = [np.nan]
                        print(f'{file} peds_{cnt:02} is none name')

                    if len(ids) > 3: 
                        dic_ped[f'peds_{cnt:02}_ID'] = [ids[3]]
                    else:
                        dic_ped[f'peds_{cnt:02}_ID'] = [np.nan]
                        print(f'{file} peds_{cnt:02} is none id')

                    # カウンタ加算
                    cnt += 1

                # データフレーム変換
                df = pd.DataFrame.from_dict(dic_ped)
                df.index = [index] * len(df)
                # 保存
                df.to_pickle(f'./html/pedigree_table/{index}.pickle')
            except Exception as e:
                print(f'{file}, {e}')
                continue

        # 全血統表に追加するデータ作成
        path = './html/pedigree_table'
        # ファイルリスト取得
        files = os.listdir(path)
        # データフレーム作成
        df = pd.DataFrame()
        # ファイルリストループ
        for file in tqdm(files):
            if '.pickle' not in file:
                print(f'Not target {file}')
                continue
            t = os.path.getctime(f'{path}/{file}')
            d = datetime.fromtimestamp(t)
            if datetime.today().date() != d.date():
                continue
                
            print(f'date match {file}')
            add_df = pd.read_pickle(f'{path}/{file}')
            df = pd.concat([df, add_df])

        df.to_pickle('./html/pedigree_table/all/add_pedigree_table.pickle')
        
        # マージ
        # 古いファイル読込
        old_df = pd.read_pickle('./html/pedigree_table/all/pedigree_table.pickle')
        # 追加ファイル読込
        add_df = pd.read_pickle('./html/pedigree_table/all/add_pedigree_table.pickle')
        # マージ
        filtered_old = old_df[~old_df.index.isin(add_df.index)]
        df = pd.concat([filtered_old, add_df])
        # index で重複削除
        df = df[~df.index.duplicated(keep='first')]
        print(f'old = {len(old_df)}, add = {len(add_df)}, df = {len(df)}')
        # 保存
        df.to_pickle('./html/pedigree_table/all/pedigree_table.pickle')

    @staticmethod    
    def preprocess_core(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """ 血統表データ前処理 
            Args:
                df: pl.DataFrame: 血統表データ
            Returns:
                pl.DataFrame: 前処理後データ
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        return df.select(pl.col("^.*(_ID|type).*$"))

    @staticmethod
    def preprocess() -> None:
        """ 血統表データ前処理 """
        pedigree_table = pd.read_pickle(PathManager.get_html_pedigree_table())
        pedigree_table['馬名_ID'] = pedigree_table.index
        pedigree_table = pl.from_pandas(pedigree_table)
        df = PedigreeTable.preprocess_core(pedigree_table)
        save_path = PathManager.get_pedigree_table(False)
        df.write_parquet(save_path)
