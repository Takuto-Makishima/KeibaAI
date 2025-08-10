import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
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


class RaceTable:
    """ レーステーブルクラス """
    HTML_COLUMNS = ['着順', '枠番', '馬番', '馬名', '馬名_ID', '性齢', '斤量', '騎手', '騎手_ID', 'タイム', '着差',
                    'ﾀｲﾑ指数', '通過', '上り', '単勝', '人気', '馬体重', '調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ', '備考', '調教師',
                    '調教師_ID', '馬主', '馬主_ID', '賞金(万円)', 'race_id', '日付', 'レース名1', 'レース名2',
                    'レースタイプ', 'レース周り', '距離', '馬場', '天気', '開催数', '会場', '開催日', 
                    '詳細条件_0', '詳細条件_1', '詳細条件_2', '詳細条件_3', '詳細条件_4', '馬場指数', 
                    'ペース', 'ラップ', 'ラップタイム', '前後半_ラップ', 
                    '1コーナー', '2コーナー', '3コーナー', '4コーナー']
    LIST = []
    DICT = {}
    def __init__(self):
        """ コンストラクタ """
        pass
    
    @staticmethod
    def scraping(driver: WebDriver, days: List[datetime] = [], day: Optional[datetime] = None, now_race_id: str = '', now_race_ids: List[str] = []) -> None:
        """ レーステーブル取得 
            Args:
                driver (WebDriver): ドライバ
                days (List[datetime]): 取得日リスト
                day (Optional[datetime]): 取得日
                now_race_id (str): レースID
                now_race_ids (List[str]): レースIDリスト
            Returns:
                None
        """
        temp = None
        race_ids = []
        lst = [datetime.strftime(i, '%Y%m%d') for i in days]
        for file in lst:
            file_path = f'./html/race_ids/{file[:4]}/{file}.bin'
            with open(file_path) as f:
                temp = f.read()

            soup = BeautifulSoup(temp, 'html.parser')
            # データリスト取得
            data = soup.find_all(class_='RaceList_DataList')
            # 取得したリストを解析
            for element in data:
                # レースID
                data_items = element.find_all(class_='RaceList_DataItem')
                for item in data_items:
                    a = item.find('a').get('href')
                    race_ids.append(re.findall(r"\d+", a)[0])

        cnt = 0
        for race_id in race_ids:
            print(race_id)
            if race_id == '':
                continue
            try:
                # ディレクトリ作成
                result_db_path = f'./html/race_result_db/{race_id[:4]}'
                if os.path.isdir(result_db_path) == False:
                    print(f'create is {result_db_path} folder')
                    os.makedirs(result_db_path)

                # 保存
                result_db_file = f'{result_db_path}/{race_id}.bin'
                if os.path.isfile(result_db_file) == False:
                    # 接続先
                    url = f'https://db.netkeiba.com/race/{race_id}/'
                    # 最大3回実行
                    for _ in range(3):
                        try:
                            driver.get(url)
                            # データ取得まで待つ
                            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'race_table_01.nk_tb_common')))
                            time.sleep(1)
                        except Exception as e:
                            print(f'db result continue: {e}')
                            continue
                        else:
                            # 失敗しなかった場合は、ループを抜ける
                            break
                    else:
                        msg = f'取得失敗:{url}'
                        raise TimeoutException(msg)
                        
                    data = BeautifulSoup(driver.page_source, 'html.parser')
                    encoded = str(data).encode('cp932', "ignore")
                    with open(result_db_file, "wb") as f:
                        f.write(encoded)

                # ディレクトリ作成
                result_path = f'./html/race_result/{race_id[:4]}'
                if os.path.isdir(result_path) == False:
                    print(f'create is {result_path} folder')
                    os.makedirs(result_path)
                
                # 保存
                result_file = f'{result_path}/{race_id}.bin'
                if os.path.isfile(result_file) == False:
                    time.sleep(1)
                    # 接続先
                    url = f'https://race.netkeiba.com/race/result.html?race_id={race_id}'
                    for _ in range(3):
                        try:
                            driver.get(url)
                            # データ取得まで待つ
                            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'ResultTableWrap')))
                            time.sleep(1)
                        except Exception as e:
                            # 失敗時の処理(不要ならpass)
                            tableElem = None
                            try:
                                tableElem = driver.find_element(by=By.CLASS_NAME, value='ResultTableWrap')
                                if tableElem != None:
                                    break
                            except NoSuchElementException:
                                print('race result continue')
                                continue
                        else:
                            # 失敗しなかった場合は、ループを抜ける
                            break
                    else:
                        msg = f'取得失敗:{url}'
                        raise TimeoutException(msg)
                        
                    data = BeautifulSoup(driver.page_source, 'html.parser')
                    encoded = str(data).encode('cp932', "ignore")
                    with open(result_file, "wb") as f:
                        f.write(encoded)

                cnt += 1
            #存在しないrace_idを飛ばす
            except IndexError:
                print(f'IndexError {race_id}')
                Notification.send(f'RaceTable scraping IndexError {race_id}: {e}')
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(race_id, e)
                Notification.send(f'RaceTable scraping Exception {race_id}: {e}')
                #error = True
                continue
            #Jupyterで停止ボタンを押した時の対処    
            except:
                error = True
                break
    
    @staticmethod
    def read_race_ids(file_path: str) -> List[str]:
        """ レースID読み込み
            Args:
                file_path (str): レースIDファイルパス
            Returns:
                List[str]: レースIDリスト
        """
        with open(file_path) as f:
            race_ids = f.read().split('\n')
        return [race_id for race_id in race_ids if race_id]

    @staticmethod
    def parse_html(file_path: str) -> BeautifulSoup:
        """ HTMLファイルをパース
            Args:
                file_path (str): HTMLファイルパス
            Returns:
                BeautifulSoup: パースされたHTML
        """
        with open(file_path) as f:
            return BeautifulSoup(f.read(), 'html.parser')
    
    @staticmethod
    def extract_race_data(soup: BeautifulSoup) -> Optional[dict]:
        """ レースデータ抽出
            Args:
                soup (BeautifulSoup): パースされたHTML
            Returns:
                Optional[dict]: 抽出されたレースデータ辞書
        """
        dic_data = {}
        dic_header = {}
        table = soup.find('table', attrs={'class': 'race_table_01'})
        if table is None:
            return None

        trs = table.find_all('tr')
        if len(dic_header) == 0:
            ths = trs[0].find_all('th')
            for col_index, th in enumerate(ths):
                dic_header[col_index] = th.text.replace('\n', '').replace(' ', '')

        for i in range(1, len(trs)):
            tds = trs[i].find_all('td')
            for col_index, col_name in dic_header.items():
                if col_name in dic_data:
                    dic_data[col_name].append(tds[col_index].text.replace('\n', '').replace(' ', ''))
                    if tds[col_index].find('a') and col_name not in ['調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ']:
                        dic_data[col_name + '_ID'].append(re.findall(r'\d+', tds[col_index].find('a').get('href'))[0])
                else:
                    dic_data[col_name] = [tds[col_index].text.replace('\n', '').replace(' ', '')]
                    if tds[col_index].find('a') and col_name not in ['調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ']:
                        dic_data[col_name + '_ID'] = [re.findall(r'\d+', tds[col_index].find('a').get('href'))[0]]
        
        return dic_data
    
    @staticmethod
    def extract_race_info(soup: BeautifulSoup) -> Optional[dict]:
        """ レース情報抽出
            Args:
                soup (BeautifulSoup): パースされたHTML
            Returns:
                Optional[dict]: 抽出されたレース情報辞書
        """
        smalltxt = soup.find("p", class_="smalltxt")
        text = smalltxt.get_text(strip=True) if smalltxt else None
        if text is None:
            return None
        text = unicodedata.normalize('NFKC', text)
        # 開催情報（「◯回」「競馬場」「◯日目」）を抽出
        kaisai_block = re.search(r'\d+回[^\d\s]+?\d+日目', text)
        kaisai = []
        if kaisai_block:
            # そこから「◯回」「競馬場名」「◯日目」を取り出す
            kaisai = re.findall(r'\d+回|[^\d\s]+?(?=\d+日目)|\d+日目', kaisai_block.group())
        # 条件情報（牝、[指]、(馬齢) など）を抽出
        joken_matches = re.findall(r'\((.*?)\)|\[(.*?)\]|(\b牝\b)', text)
        joken = [m for group in joken_matches for m in group if m]

        pattern = r'(?:[2-4]歳(?:以上)?)\s*(?:(?:新馬|未勝利|オープン)|\d+勝クラス|\d+万下)'
        match = re.search(pattern, text)
        condition = match.group()

        dic = {
            '開催数': kaisai[0] if len(kaisai) > 0 else "",
            '会場': kaisai[1] if len(kaisai) > 1 else "",
            '開催日': kaisai[2] if len(kaisai) > 2 else "",
            }
        dic['詳細条件_0'] = condition
        for i, data in enumerate(joken):
            dic[f'詳細条件_{i+1}'] = data
        return dic

    @staticmethod
    def extract_date(soup: BeautifulSoup) -> Optional[datetime.date]:
        """ レース日付抽出
            Args:
                soup (BeautifulSoup): パースされたHTML
            Returns:
                Optional[datetime.date]: 抽出されたレース日付
        """
        txt = soup.find(class_='smalltxt')
        if txt:
            date_match = re.findall(r'(\d+)年(\d+)月(\d+)日', txt.text.replace('\n', ''))
            if date_match:
                return datetime(int(date_match[0][0]), int(date_match[0][1]), int(date_match[0][2])).date()
        return None

    @staticmethod
    def extract_ground_index(soup: BeautifulSoup) -> Optional[str]:
        """ 馬場指数抽出
            Args:
                soup (BeautifulSoup): パースされたHTML
            Returns:
                Optional[str]: 抽出された馬場指数
        """
        table = soup.find('table', attrs={'class': 'result_table_02'})
        if table:
            tr = table.find_all('tr')[0]
            th = tr.find('th')
            if th and '馬場指数' in th.text:
                td = tr.find('td')
                return re.findall(r'\d+', td.text.replace(r'\n', '').replace(' ', ''))[0]
        return None

    @staticmethod
    def extract_lap_times(soup: BeautifulSoup) -> tuple:
        """ ラップタイム抽出
            Args:
                soup (BeautifulSoup): パースされたHTML
            Returns:
                tuple: ラップ、ラップタイム、前後半ラップ
        """
        rap = ""
        rap_time = ""
        half_rap = ""
        
        table = soup.find("table", class_="result_table_02", summary=lambda x: x and "ラップタイム" in x)
        if table:
            for row in table.find_all("tr"):
                th = row.find("th")
                td = row.find("td", class_="race_lap_cell")
                if th and "ペース" in th.text and td:
                    aaa = td.text.strip().split("\xa0")
                    if len(aaa) >= 2:
                        rap_time = aaa[0]
                        half_rap = aaa[1].replace('(', '').replace(')', '')
                elif th and "ラップ" in th.text and td:
                    rap = td.text.strip()
        
        return rap, rap_time, half_rap
    
    @staticmethod
    def extract_expand(soup: BeautifulSoup) -> dict:
        """ 拡張情報抽出
            Args:
                soup (BeautifulSoup): パースされたHTML
            Returns:
                dict: 抽出された拡張情報辞書
        """
        dic = {}
        table = soup.find("table", class_="result_table_02", summary=lambda x: x and "コーナー通過順位" in x)
        if table:
            for row in table.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if td:
                    dic[th.text.strip()] = td.text.strip()
        return dic

    @staticmethod
    def create_race_info(date: datetime.date, race_name1: str, ground_index: str, rap: str, rap_time: str, half_rap: str, expand: dict, info: dict) -> dict:
        """ レース情報作成
            Args:
                date (datetime.date): レース日付
                race_name1 (str): レース名1
                ground_index (str): 馬場指数
                rap (str): ラップ
                rap_time (str): ラップタイム
                half_rap (str): 前後半ラップ
                expand (dict): 拡張情報
                info (dict): レース情報
            Returns:
                dict: 作成されたレース情報辞書
        """
        race_info = {
            '日付': date, 'レース名1': race_name1, "レース名2": "", 
            "レースタイプ": "", "レース周り": "", "距離": "",
            "馬場": "", "天気": "", '馬場指数': ground_index,"ペース": "",
            'ラップ': rap, 'ラップタイム': rap_time, '前後半_ラップ': half_rap
        }
        for key, value in expand.items():
            race_info[key] = value

        for key, value in info.items():
            race_info[key] = value

        return race_info

    @staticmethod
    def extract_race_data2(soup: BeautifulSoup, race_id: str, dic_data: dict, race_info: dict) -> pd.DataFrame:
        """ レースデータ抽出2 
            Args:
                soup (BeautifulSoup): パースされたHTML
                race_id (str): レースID
                dic_data (dict): レースデータ辞書
                race_info (dict): レース情報辞書
            Returns:
                pd.DataFrame: 抽出されたレースデータのDataFrame
        """
        race_data = soup.find(class_='RaceList_Item02')
        if race_data:
            if name := race_data.find(class_='RaceName'):
                race_info["レース名2"] = name.get_text(strip=True)
            race_data_1 = race_data.find(class_='RaceData01')
            if race_data_1:
                for info in race_data_1:
                    texts = re.findall(r'\w+', info.text)
                    for text in texts:
                        if not race_info["レースタイプ"]:
                            if '芝' in text:
                                race_info["レースタイプ"] = '芝'
                            elif 'ダ' in text:
                                race_info["レースタイプ"] = 'ダート'
                            elif '障' in text:
                                race_info["レースタイプ"] = '障害'
                        if not race_info["レース周り"]:
                            race_info["レース周り"] = next((t for t in ['右', '左', '直線'] if t in text), "")
                        if not race_info["距離"] and 'm' in text:
                            race_info["距離"] = re.findall(r'\d+', text)[0]
                        if not race_info["馬場"]:
                            race_info["馬場"] = {"良": "良", "稍": "稍重", "重": "重", "不": "不良"}.get(text, "")
                        if not race_info["天気"] and text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                            race_info["天気"] = text

        if div := soup.find('div', class_='RapPace_Title'):
            if span := div.find('span'):
                race_info["ペース"] = span.text.strip().replace('\n', '').replace(' ', '')
        
        df_add = pd.DataFrame.from_dict(dic_data)
        df_add.index = [race_id] * len(df_add)
        df_add["race_id"] = df_add.index
        for key, value in race_info.items():
            df_add[key] = value
        df1_only_columns = set(df_add.columns) - set(RaceTable.HTML_COLUMNS)
        if df1_only_columns:
            RaceTable.DICT[race_id] = list(df1_only_columns)
        return df_add

    @staticmethod
    def create_data(start: int, end: int) -> None:
        """ データ作成
            Args:
                start (int): 開始年
                end (int): 終了年
            Returns:
                None
        """
        RaceTable.LIST = []
        RaceTable.DICT = {}
        for year in range(start, end):
            race_ids_path = f'./html/race_ids/{year}'
            result_db_path = f'./html/race_result_db/{year}'
            result_path = f'./html/race_result/{year}'
            # レースID取得
            race_ids = RaceTable.read_race_ids(f'{race_ids_path}/race_ids_{year}.txt')
            df = pd.DataFrame()
            for race_id in tqdm(race_ids):
                try:
                    result_db_html = RaceTable.parse_html(f'{result_db_path}/{race_id}.bin')
                    dic_data = RaceTable.extract_race_data(result_db_html)
                    if not dic_data:
                        RaceTable.LIST.append(race_id)
                        continue
                    date = RaceTable.extract_date(result_db_html)
                    ground_index = RaceTable.extract_ground_index(result_db_html)
                    race_name1 = result_db_html.find(class_='data_intro').find('h1').get_text(strip=True) if result_db_html.find(class_='data_intro') else ''
                    rap, rap_time, half_rap = RaceTable.extract_lap_times(result_db_html)
                    expand = RaceTable.extract_expand(result_db_html)
                    info = RaceTable.extract_race_info(result_db_html)
                    if info is None:
                        RaceTable.LIST.append(race_id)
                        continue                   
                    result_html = RaceTable.parse_html(f'{result_path}/{race_id}.bin')
                    race_info = RaceTable.create_race_info(date, race_name1, ground_index, rap, rap_time, half_rap, expand, info)
                    df_add = RaceTable.extract_race_data2(result_html, race_id, dic_data, race_info)
                    df = pd.concat([df,df_add])
                #wifiの接続が切れた時などでも途中までのデータを返せるようにする
                except Exception as e:
                    print(f'Exception {race_id} {e}')
                    continue
                #Jupyterで停止ボタンを押した時の対処    
                except:
                    print(f'except {race_id}')
                    break
            race_table_path = f'./html/race_table'
            if os.path.isdir(race_table_path) == False:
                print(f'create is {race_table_path} folder')
                os.makedirs(race_table_path)
            race_table_file_path = PathManager.get_html_race_table(year, False)
            df = pl.from_pandas(df)
            df.write_parquet(race_table_file_path)
        print("Contains None:", RaceTable.LIST)
        print("Unnecessary columns:", RaceTable.DICT)

    @staticmethod
    def cnv_order(x) -> int:
        """ 着順変換 
            Args:
                x: 着順
            Returns:
                int: 着順
        """
        tmp = str(x)
        if '中' in tmp:
            return 20
        elif '失' in tmp:
            return 21
        elif '除' in tmp:
            return 22
        elif '取' in tmp:
            return 23
    
        matches = re.findall(r'\(.*?\)', tmp)
        if len(matches) != 0:
            tmp = tmp.replace(matches[0], '')
            return int(tmp)
        else:
            return int(tmp)

    @staticmethod
    def cnv_time(x: str) -> float:
        """ 時間変換
            Args:
                x (str): 時間
            Returns:
                float: 時間
        """
        if x == '':
            return None
        exc = re.findall(r'\d+', str(x))
        if len(exc) == 3:
            millisec = exc[2].ljust(6, '0')[:6]
            temp = '00:{0:0>2}:{1:0>2}.{2}'.format(exc[0], exc[1], millisec)
        elif len(exc) == 2:
            millisec = exc[1].ljust(6, '0')[:6]
            temp = '00:00:{0:0>2}.{1}'.format(exc[0], millisec)
        else:
            return None
        
        # `temp` を datetime.strptime でパースし、timedelta に変換
        dt = datetime.strptime(temp, "%H:%M:%S.%f")
        delta = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)

        return delta.total_seconds()

    @staticmethod
    def cnv_sex(x: str) -> str:
        """ 性別変換
            Args:
                x (str): 性別
            Returns:
                str: 性別
        """
        if '牡' in x:
            return '牡'
        elif '牝' in x:
            return '牝'
        elif 'セ' in x:
            return 'セ'
        else:
            return np.nan

    @staticmethod
    def cnv_age(x) -> int:
        """ 年齢変換
            Args:
                x: 年齢
            Returns:
                int: 年齢
        """
        temp = str(x)[1:]
        return int(temp)

    @staticmethod
    def cnv_affiliation(x: str) -> str:
        """ 所属変換
            Args:
                x (str): 所属
            Returns:
                str: 所属
        """
        temp = x.replace('[', '').split(']')
        if len(temp) != 0:
            return temp[0]
        else:
            return np.nan

    @staticmethod
    def cnv_trainer(x: str) -> str:
        """ 調教師変換
            Args:
                x (str): 調教師
            Returns:
                str: 調教師
        """
        temp = x.replace('[', '').split(']')
        if len(temp) != 0:
            return temp[-1]
        else:
            return np.nan

    @staticmethod
    def cnv_first_passing(x: str) -> float:
        """ 1角通過変換
            Args:
                x (str): 1角通過
            Returns:
                float: 1角通過
        """
        if x=='':
            return None
        temp = str(x).split('-')
        return float(temp[0])

    @staticmethod
    def cnv_latter_passing(x: str) -> float:
        """ 4角通過変換
            Args:
                x (str): 4角通過
            Returns:
                float: 4角通過
        """
        if x=='':
            return None
        temp = str(x).split('-')
        return float(temp[-1])

    @staticmethod
    def cnv_position_type(passing: str, num_horses: int) -> str:
        if passing == '':
            return None
        positions = list(map(int, passing.split("-")))  # 各コーナーの通過順位をリスト化
        last_corner_pos = positions[-1]  # 最終コーナーの順位
        other_corners_pos = positions[:-1]  # 最終コーナー以外の順位

        # 逃げ: 最終コーナー以外で1位がある場合
        if any(pos == 1 for pos in other_corners_pos):
            return "逃げ"

        # 先行: 逃げでなく、最終コーナー4位以内
        if last_corner_pos <= 4:
            return "先行"

        # 差し: 逃げ・先行でなく、最終コーナーの順位が出走頭数の3分の2以内
        if num_horses >= 8 and last_corner_pos <= (num_horses * 2 // 3):
            return "差し"

        # 追込: 上記以外
        return "追込"
    
    @staticmethod
    def cnv_horse_weight_change(x) -> int:
        """ 馬体重増減変換
            Args:
                x: 馬体重変化
            Returns:
                int: 馬体重変化
        """
        # x が None または空文字の場合
        if x is None or str(x).strip() == '':
            return 0
        # x が特定の無効値の場合
        if any(invalid in str(x) for invalid in ['--', '計不', '前計不']):
            return 0
        try:
            # 括弧内の数値を抽出して変換
            temp = str(x).split('(')[-1].replace(')', '')
            return int(temp)
        except (ValueError, IndexError):
            # 数値変換に失敗した場合は 0 を返す
            return 0
    
    @staticmethod
    def cnv_horse_weight(x) -> int:
        """ 馬体重変化
            Args:
                x: 馬体重
            Returns:
                int: 馬体重
        """
        # x が None または空文字の場合
        if x is None or str(x).strip() == '':
            return 0
        # x が特定の無効値の場合
        if any(invalid in str(x) for invalid in ['--', '計不', '前計不']):
            return 0
        try:
            # 括弧内の数値を抽出して変換
            temp = str(x).split('(')[0]
            return int(temp)
        except (ValueError, IndexError):
            # 数値変換に失敗した場合は 0 を返す
            return 0

    @staticmethod
    def cnv_class(x) -> str:
        """ クラス変換
            Args:
                x: クラス
            Returns:
                str: クラス
        """
        if '新馬' in x['詳細条件_0'] or '新馬' in x['レース名1']:
            return '新馬クラス'
        elif '未勝利' in x['詳細条件_0'] or '未勝利' in x['レース名1']:
            return '未勝利クラス'
        elif '１勝' in x['詳細条件_0'] or '1勝' in x['詳細条件_0'] or '500万下' in x['詳細条件_0'] or '５００万下' in x['詳細条件_0'] or '500万下' in x['レース名1']:
            return '1勝クラス'
        elif '２勝' in x['詳細条件_0'] or '2勝' in x['詳細条件_0'] or '1000万下' in x['詳細条件_0'] or '１０００万下' in x['詳細条件_0'] or '1000万下' in x['レース名1']:
            return '2勝クラス'
        elif '３勝' in x['詳細条件_0'] or '3勝' in x['詳細条件_0'] or '1600万下' in x['詳細条件_0'] or '１６００万下' in x['詳細条件_0'] or '1600万下' in x['レース名1']:
            return '3勝クラス'
        elif 'オープン' in x['詳細条件_0']:
            if 'G3' in x['レース名1']:
                return 'G3クラス'
            elif 'G2' in x['レース名1']:
                return 'G2クラス'
            elif 'G1' in x['レース名1']:
                return 'G1クラス'
            else:
                return 'オープンクラス'
        else:
            return np.nan

    @staticmethod
    def cnv_age_4_over_condition(x) -> int:
        """ 4歳以上変換
            Args:
                x: データ
            Returns:
                int: 1=4歳以上
        """
        if '４歳' in x['詳細条件_0'] or '4歳' in x['詳細条件_0'] or '4歳' in x['レース名1'] or '４歳' in x['レース名1']:
            return 1
        else:
            return 0

    @staticmethod
    def cnv_age_3_over_condition(x) -> int:
        """ 3歳以上変換
            Args:
                x: データ
            Returns:
                int: 1=3歳以上
        """
        if '３歳以上' in x['詳細条件_0'] or '3歳以上' in x['詳細条件_0'] or '３歳以上' in x['レース名1'] or '3歳以上' in x['レース名1']:
            return 1
        else:
            return 0

    @staticmethod
    def cnv_age_3_condition(x) -> int:
        """ 3歳変換
            Args:
                x: データ
            Returns:
                int: 1=3歳
        """
        if '３歳' in x['詳細条件_0'] and '以上' not in x['詳細条件_0']:
            return 1
        elif '3歳' in x['詳細条件_0'] and '以上' not in x['詳細条件_0']:
            return 1
        elif '３歳' in x['レース名1'] and '以上' not in x['レース名1']:
            return 1
        elif '3歳' in x['レース名1'] and '以上' not in x['レース名1']:
            return 1
        else:
            return 0

    @staticmethod
    def cnv_age_2_condition(x) -> int:
        """ 2歳変換
            Args:
                x: データ
            Returns:
                int: 1=2歳
        """
        if '２歳'in x['詳細条件_0'] or '2歳' in x['詳細条件_0'] or '２歳'in x['レース名1'] or '2歳' in x['レース名1']:
            return 1
        else:
            return 0

    @staticmethod
    def same_generation_only(x) -> int:
        """ 世代限定変換
            Args:
                x: データ
            Returns:
                int: 1=世代限定
        """
        if '３歳' in x['詳細条件_0'] and '以上' not in x['詳細条件_0']:
            return 1
        elif '3歳' in x['詳細条件_0'] and '以上' not in x['詳細条件_0']:
            return 1
        elif '３歳' in x['レース名1'] and '以上' not in x['レース名1']:
            return 1
        elif '3歳' in x['レース名1'] and '以上' not in x['レース名1']:
            return 1
        elif '２歳'in x['詳細条件_0'] or '2歳' in x['詳細条件_0']:
            return 1        
        elif '新馬'in x['詳細条件_0'] or '新馬' in x['レース名1']:
            return 1
        else:
            return 0

    @staticmethod
    def cnv_age_condition(x) -> str:
        """ 年齢条件変換
            Args:
                x: データ
            Returns:
                str: 年齢条件
        """
        if x['4歳以上'] == 1 or '4歳' in x['レース名'] or '４歳' in x['レース名']:
            return '4歳以上'
        elif x['3歳以上'] == 1 or '3歳以上' in x['レース名']:
            return '3歳以上'
        elif x['3歳'] == 1 or '3歳' in x['レース名']:
            return '3歳'
        elif x['2歳'] == 1 or '2歳' in x['レース名']:
            return '2歳'
        else:
            return np.nan

    @staticmethod
    def cnv_affiliation_pred(x: str) -> str:
        """ 所属変換
            Args:
                x (str): 所属
            Returns:
                str: 所属
        """
        if x == '美浦':
            return '東'
        elif x == '栗東':
            return '西'
        elif '地' in x:
            return '地'
        elif '外' in x:
            return '外'
        
    @staticmethod
    def cnv_weight(x) -> float:
        """ 斤量変換
            Args:
                x: 斤量
            Returns:
                float: 斤量
        """
        if np.isnan(x['1走前_斤量']) or np.isnan(x['斤量']):
            return np.nan
        return float(x['斤量']) - float(x['1走前_斤量'])

    @staticmethod
    def cnv_distance(x) -> int:
        """ 距離変換
            Args:
                x: データ
            Returns:
                int: 距離
        """
        if np.isnan(x['1走前_距離']) or np.isnan(x['距離']):
            return np.nan
        return int(x['距離']) - int(x['1走前_距離'])

    @staticmethod
    def cnv_smile(x: int) -> str:
        """ SMILE変換
            Args:
                x: データ
            Returns:
                str: SMILE
        """
        if x <= 1300:
            return 'S'
        elif x <= 1899:
            return 'M'
        elif x <= 2100:
            return 'I'
        elif x <= 2700:
            return 'L'
        else:
            return 'E'

    @staticmethod
    def cnv_classify_pace(pace: str, lap_str: str) -> str:

        if lap_str == '':
            return '不明'
        # ラップを float のリストに変換
        laps = list(map(float, lap_str.strip('[]').split(' - ')))
        
        if len(laps) < 4:
            return '不明'

        if pace == 'S':
            # 右肩上がり型：後半頭（4F目以降）から徐々にペースアップ
            # → 前半(1~3F)の平均 > 後半(4F~last)の平均 かつ、ラスト3Fが連続で速くなる
            front_avg = np.mean(laps[:3])
            back_avg = np.mean(laps[3:])
            last3 = laps[-3:]
            if front_avg > back_avg and last3[0] > last3[1] and last3[1] > last3[2]:
                return '右肩上がり型'
            elif last3[0] > last3[1] and last3[1] > last3[2]:
                return 'ヨーイドン型'
            else:
                return 'その他S型'
        elif pace == 'H':
            # 直線手前スロー型：後半中盤で息が入り、ラスト3Fで再加速
            # 一本調子型：途中で緩まない（前半〜後半まで一貫して速い）
            last3 = laps[-3:]
            before_last3 = laps[:-3]
            min_before_last3 = min(before_last3) if before_last3 else float('inf')
            if min_before_last3 < last3[0] and last3[0] > last3[1] > last3[2]:
                return '直線手前スロー型'
            else:
                return '一本調子型'
        elif pace == 'M':
            # 初期2Fを除いた区間の max - min をチェック
            middle_section = laps[2:]
            if len(middle_section) < 2:
                return '不明'
            lap_range = max(middle_section) - min(middle_section)
            if lap_range >= 1.0:
                return '緩急型'
            else:
                return '一定型'

        return '不明'

    @staticmethod
    def get_date_list(date_list) -> List[List[datetime]]:
        """ 連続している日付のグループを取得
            Args:
                date_list: 日付リスト
            Returns:
                List[List[datetime]]: 連続している日付のグループ
        """
        # 連続している日付のグループを格納するリスト
        consecutive_groups = []
        
        # 最初の日付をグループの始まりとして設定
        start_date = date_list[0]
        current_group = [start_date]
        
        # 日付リストを反復処理
        for i in range(1, len(date_list)):
            if (date_list[i] - date_list[i - 1]).days == 1:
                # 日付が連続している場合、グループに追加
                current_group.append(date_list[i])
            else:
                # 日付が連続していない場合、現在のグループをリストに追加し、新しいグループを開始
                consecutive_groups.append(current_group)
                current_group = [date_list[i]]
        
        # 最後のグループをリストに追加
        consecutive_groups.append(current_group)
        
        return consecutive_groups
    
    @staticmethod
    def preprocess_core(data_frame: Union[pd.DataFrame,pl.DataFrame], is_pred: bool=False) -> pl.DataFrame:
        """ 前処理コア
            Args:
                data_frame: データフレーム
                is_pred: 予測データか
            Returns:
                data_frame: データフレーム
        """
        if isinstance(data_frame, pd.DataFrame):
            df = pl.from_pandas(data_frame)
        else:
            df = data_frame.clone()
        df = df.with_columns([
            #pl.when(df.schema["日付"] != pl.Date).then(pl.col("日付").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d")).otherwise(pl.col("日付")).alias("日付"),
            pl.when(df.schema["日付"] != pl.Date).then(pl.col("日付").cast(pl.Date)).otherwise(pl.col("日付")).alias("日付"),
            pl.col('枠番').count().over('race_id').alias('頭数'),
            pl.col('race_id').str.slice(4, 2).cast(pl.Int32).alias('会場_ID'),
            pl.col('race_id').str.slice(6, 2).cast(pl.Int32).alias('開催数_ID'),
            pl.col('race_id').str.slice(8, 2).cast(pl.Int32).alias('開催日_ID'),
            pl.col('race_id').str.slice(10, 2).cast(pl.Int32).alias('レース番号'),
            pl.col('枠番').cast(pl.Int32),
            pl.col('馬番').cast(pl.Int32),
            pl.col('距離').cast(pl.Int32),
            pl.struct('詳細条件_0', 'レース名1').map_elements(lambda x: RaceTable.cnv_class(x), return_dtype=pl.Utf8).alias('クラス'),
            pl.col('性齢').map_elements(lambda x: RaceTable.cnv_sex(x), return_dtype=pl.Utf8).alias('性別'),
            pl.col('性齢').map_elements(lambda x: RaceTable.cnv_age(x), return_dtype=pl.Int32).alias('年齢'),
            pl.col('馬体重').map_elements(lambda x: RaceTable.cnv_horse_weight_change(x), return_dtype=pl.Int32).alias('馬体重増減'),
            pl.col('馬体重').map_elements(lambda x: RaceTable.cnv_horse_weight(x), return_dtype=pl.Int32).alias('馬体重'),
            pl.col('単勝').map_elements(lambda x: None if '-' in x else float(x), return_dtype=pl.Float64).alias('単勝'),
            pl.col('人気').map_elements(lambda x: None if x == '' else float(x), return_dtype=pl.Float64).cast(pl.Int32).alias('人気'),
            pl.col('距離').cast(pl.Int32).map_elements(lambda x: RaceTable.cnv_smile(x), return_dtype=pl.Utf8).alias('SMILE'),
        ])
        # 0 / 1 に変換
        df = df.with_columns([
            pl.struct('詳細条件_0', 'レース名1').map_elements(lambda x: RaceTable.cnv_age_4_over_condition(x), return_dtype=pl.Int32).alias('4歳以上'),
            pl.struct('詳細条件_0', 'レース名1').map_elements(lambda x: RaceTable.cnv_age_3_over_condition(x), return_dtype=pl.Int32).alias('3歳以上'),
            pl.struct('詳細条件_0', 'レース名1').map_elements(lambda x: RaceTable.cnv_age_3_condition(x), return_dtype=pl.Int32).alias('3歳'),
            pl.struct('詳細条件_0', 'レース名1').map_elements(lambda x: RaceTable.cnv_age_2_condition(x), return_dtype=pl.Int32).alias('2歳'),
            pl.struct('詳細条件_0', 'レース名1').map_elements(lambda x: RaceTable.same_generation_only(x), return_dtype=pl.Int32).alias('同世代限定')
        ])
        for value in ['馬齢', '指', '特指', '定量', '別定', 'ハンデ', '国際', '牡', '牝', '混', '見習騎手', '九州産馬']:
            df = df.with_columns([
                pl.lit(0).alias(value)
            ])
            for col in ['詳細条件_1', '詳細条件_2', '詳細条件_3', '詳細条件_4']:
                df = df.with_columns([
                    pl.when(pl.col(col) == value).then(1).otherwise(pl.col(value)).alias(value)
                ])
        if not is_pred:
            df = df.with_columns([
                pl.col('着順').map_elements(lambda x: RaceTable.cnv_order(x), pl.Int32).cast(pl.Int32).alias('着順'),
                pl.col('調教師').map_elements(lambda x: RaceTable.cnv_affiliation(x), pl.Utf8).alias('所属'),
                pl.col('調教師').map_elements(lambda x: RaceTable.cnv_trainer(x), pl.Utf8).alias('調教師'),
                pl.col('通過').map_elements(lambda x: RaceTable.cnv_first_passing(x), pl.Float64).alias('1角'),
                pl.col('通過').map_elements(lambda x: RaceTable.cnv_latter_passing(x), pl.Float64).alias('4角'),
                pl.struct(["通過", "頭数"]).map_elements(lambda row: RaceTable.cnv_position_type(row["通過"], row["頭数"]), pl.Utf8).alias("脚質")
            ])
            df = df.with_columns([
                (pl.col('1角') / pl.col('頭数')).alias('位置取り'),
            ])
            df = df.with_columns([
                pl.col('タイム').map_elements(lambda x: RaceTable.cnv_time(x), pl.Float64).alias('タイム'),
                pl.col('賞金(万円)').map_elements(lambda x: None if x == '' else float(x.replace(',', '')), pl.Float64).alias('賞金(万円)'),
                pl.col('上り').map_elements(lambda x: None if x == '' else float(x), pl.Float64).alias('上り'),
                pl.col('ﾀｲﾑ指数').map_elements(lambda x: None if x == '' else float(x), pl.Float64).alias('ﾀｲﾑ指数')
            ])
            df = df.with_columns([
                pl.struct(['ペース', 'ラップ']).map_elements(lambda row: RaceTable.cnv_classify_pace(row["ペース"], row["ラップ"]), return_dtype=pl.Utf8).alias('ペース詳細')
            ])
            df = df.with_columns([
                (pl.col("着順") == 1).cast(pl.Int8).alias("top1"),
                ((pl.col("着順") < 3) & (pl.col("着順") != 0)).cast(pl.Int8).alias("top2"),
                ((pl.col("着順") < 4) & (pl.col("着順") != 0)).cast(pl.Int8).alias("top3"),
                ((pl.col("着順") < 5) & (pl.col("着順") != 0)).cast(pl.Int8).alias("top4"),
                ((pl.col("着順") < 6) & (pl.col("着順") != 0)).cast(pl.Int8).alias("top5"),
                pl.when(pl.col("着順") < 2).then((1.0 / pl.col("着順") * 30).cast(pl.Int32)).otherwise(0).alias("rank1"),
                pl.when(pl.col("着順") < 3).then((1.0 / pl.col("着順") * 30).cast(pl.Int32)).otherwise(0).alias("rank2"),
                pl.when(pl.col("着順") < 4).then((1.0 / pl.col("着順") * 30).cast(pl.Int32)).otherwise(0).alias("rank3"),
                pl.when(pl.col("着順") < 5).then((1.0 / pl.col("着順") * 30).cast(pl.Int32)).otherwise(0).alias("rank4"),
                pl.when(pl.col("着順") < 6).then((1.0 / pl.col("着順") * 30).cast(pl.Int32)).otherwise(0).alias("rank5"),
            ])
            df = df.with_columns([
                pl.when(pl.col("備考").str.contains("出遅れ")).then(1).otherwise(0).alias("出遅れ"),
                pl.when(pl.col("備考").str.contains("不利")).then(1).otherwise(0).alias("不利")
            ])
            # 列名変換
            df = df.rename({
                '着順': 'order',
                '賞金(万円)': '賞金',
                'レース名2': 'レース名'
            })
            # 年齢条件
            df = df.with_columns([
                pl.struct(["4歳以上", "3歳以上", "3歳", "2歳", "レース名"]).map_elements(lambda x: RaceTable.cnv_age_condition(x), pl.Utf8).alias("年齢条件")
            ])
            # race_id昇順, 着順昇順, 人気降順
            result = (
                df.sort(["race_id", "order", "人気"], descending=[False, False, True])
                .group_by("race_id")
                .agg(
                    pl.col("人気").head(3).cast(str).str.zfill(2).str.concat(delimiter="").cast(pl.Int32).alias("荒れ指数")
                )
            )
            df = df.join(result, on="race_id", how="left")
            # 列並び替え
            df = df.select([
                'race_id', '枠番', '馬番', '馬名', '馬名_ID', '性別', '年齢', '斤量', '騎手', '騎手_ID', 
                '荒れ指数', 'タイム', '着差', 'ﾀｲﾑ指数', '1角', '4角', '位置取り', '脚質',
                '上り', '単勝', '人気', '馬体重', '馬体重増減', '備考', '所属', '調教師', '調教師_ID', '馬主', '馬主_ID', '賞金', '日付', 'レース名',
                'レースタイプ', 'レース周り', '距離', '馬場', '馬場指数', '天気', '開催数', '開催数_ID', '会場', '会場_ID', '開催日', '開催日_ID',
                'レース番号', 'クラス', '頭数', '詳細条件_0', '詳細条件_1', '詳細条件_2', '詳細条件_3', '詳細条件_4', 'ペース', 'ペース詳細',
                '4歳以上', '3歳以上', '3歳', '2歳', '同世代限定', '年齢条件',
                '馬齢', '指', '特指', '定量', '別定', 'ハンデ', '国際', '牡', '牝', '混', '見習騎手', '九州産馬',
                '1コーナー', '2コーナー', '3コーナー', '4コーナー', 'ラップ', 'ラップタイム', '前後半_ラップ', "出遅れ", "不利",
                'order', 'top1','top2','top3','top4','top5','rank1','rank2','rank3','rank4','rank5'
            ])
        else:
            df = df.with_columns([
                pl.col('所属').map_elements(lambda x: RaceTable.cnv_affiliation_pred(x), return_dtype=pl.Utf8).alias('所属'),
                pl.struct(["4歳以上", "3歳以上", "3歳", "2歳", "レース名"]).map_elements(lambda x: RaceTable.cnv_age_condition(x), pl.Utf8).alias("年齢条件")
            ])
            # 列並び替え
            df = df.select([
                'race_id', '枠番', '馬番', '馬名', '馬名_ID', '性別', '年齢', '斤量', '騎手', '騎手_ID',
                #'タイム', '着差', 'ﾀｲﾑ指数', '1角', '4角', '位置取り', '脚質','上り',
                '単勝', '人気', '馬体重', '馬体重増減',
                #'備考', 
                '所属', '調教師', '調教師_ID', '馬主', '馬主_ID',
                #'賞金',
                '日付', 'レース名',
                'レースタイプ', 'レース周り', '距離', '馬場',
                #'馬場指数',
                '天気', '開催数', '開催数_ID', '会場', '会場_ID', '開催日', '開催日_ID','レース番号',
                'クラス', '頭数', '詳細条件_0', '詳細条件_1', '詳細条件_2', '詳細条件_3', '詳細条件_4',
                #'ペース',
                '4歳以上', '3歳以上', '3歳', '2歳', '同世代限定', '年齢条件',
                '馬齢', '指', '特指', '定量', '別定', 'ハンデ', '国際', '牡', '牝', '混', '見習騎手', '九州産馬'
                #'order', 'top1','top2','top3','top4','top5','rank1','rank2','rank3','rank4','rank5'
            ])
        df = df.sort(["日付", "会場_ID", "開催数_ID", "開催日_ID", "レース番号", "馬番"])
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
            df = RaceTable.preprocess_core(pl.read_parquet(PathManager.get_html_race_table(year, False)))
            save_path = PathManager.get_race_table_extra(year, False)
            df.write_parquet(save_path)

    @staticmethod
    def merge_preprocess_data(start: int, end: int):
        """ 前処理マージ
            Args:
                start (int): 開始年
                end (int): 終了年
            Returns:
                None
        """
        read_path = PathManager.get_pedigree_table(False)
        pedigree_table = pl.read_parquet(read_path)
        
        for year in tqdm(range(start, end)):
            read_path = PathManager.get_race_table_extra(year, False)
            df = pl.read_parquet(read_path)
            race_table = df.join(pedigree_table, left_on="馬名_ID", right_on="馬名_ID", how="left")
            
            read_path = PathManager.get_training_table_extra(year, False)
            training_table = pl.read_parquet(read_path)

            merge_df = race_table.join(
                training_table,
                on=["race_id", "馬名_ID", "枠番", "馬番", "馬名"],
                how="left"
            )

            save_path=PathManager.get_merge_table_extra(year, False)
            merge_df.write_parquet(save_path)

