import os
import re
import time
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from typing import List, Optional


class RaceId:
    """ レースID """

    @staticmethod
    def scraping(driver: WebDriver, days: List[datetime] = [], day: Optional[datetime] = None, now_race_id: str = '', now_race_ids: List[str] = []) -> None:
        """ レースID取得 
            Args:
                driver (WebDriver): ドライバ
                days (List[datetime]): 取得日リスト
                day (Optional[datetime]): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (List[str]): 現在レースIDリスト
            Returns:
                None
        """
        lst = [datetime.strftime(i, '%Y%m%d') for i in days]

        for date in lst:
            # ディレクトリ作成
            dir_path = f'./html/race_ids/{date[:4]}'
            if os.path.isdir(dir_path) == False:
                print(f'create is {dir_path} folder')
                os.makedirs(dir_path)

            # ファイル存在確認
            if os.path.isfile(f'{dir_path}/{date}.bin') == True:
                continue

            # 最大3回実行
            for _ in range(3):
                try:
                    # レースID取得
                    url = 'https://race.netkeiba.com/top/race_list.html?kaisai_date=' + date
                    # ページ取得
                    driver.get(url)
                    # データ取得まで待つ
                    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'RaceList_DataList')))
                    time.sleep(1)
                except Exception as e:
                    # 失敗時の処理(不要ならpass)
                    tableElem = None
                    try:
                        tableElem = driver.find_element(by=By.CLASS_NAME, value='RaceList_DataList')
                        if tableElem != None:
                            break
                    except NoSuchElementException:
                        print('get page continue')
                        continue
                else:
                    # 失敗しなかった場合は、ループを抜ける
                    break
            else:
                msg = f'取得失敗:{url}'
                raise TimeoutException(msg)

            # html 保存
            data = BeautifulSoup(driver.page_source, 'html.parser')
            encoded = str(data).encode('cp932', "ignore")
            with open(f'{dir_path}/{date}.bin', "wb") as f:
                f.write(encoded)

    @staticmethod
    def create_data(start_year: int, end_year: int) -> None:
        """ データ作成
            Args:
                start_year (int): 開始年
                end_year (int): 終了年
            Returns:
                None
        """
        for year in range(start_year, end_year):
            path = f'./html/race_ids/{year}'            
            files = os.listdir(path)
            race_ids = []
            print(year)
            for file in tqdm(files):
                # 拡張子確認
                if '.bin' not in file:
                    print(f'Not applicable {file}')
                    continue
                # ファイル読込
                result_db_html = None
                with open(f'{path}/{file}') as f:
                    result_db_html = f.read()
                # インスタンス生成
                soup = BeautifulSoup(result_db_html, 'html.parser')
                # レースリスト取得
                lists = soup.find_all(class_='RaceList_DataItem')
                # レース数チェック
                if len(lists) != 24 and len(lists) != 36:
                    print(f'error {file} len = {len(lists)}')
                # レースID取得
                for item in lists:
                    # a タグから href 取得
                    race_id = item.find('a').get('href')
                    # 正規表現で数値のみ取得
                    race_id = re.findall(r"\d+", race_id)[0]
                    # フォーマット確認
                    if len(race_id) != 12:
                        print(f'error format {race_id}')
                        continue
                    # 追加
                    race_ids.append(race_id)
            print(f'{year} count {len(race_ids)}')
            # ファイル出力
            with open(f'{path}/race_ids_{year}.txt', 'w') as f:
                for race_id in race_ids:
                    f.write(f'{race_id}\n')
