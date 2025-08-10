import os
import re
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from datetime import datetime, timedelta
from src.utils.notification import Notification
from src.creators.dataset import Dataset


class TodayData:
    """ 今日のデータを作成するクラス """
    exec_dict = None
    exists_win5 = False

    @staticmethod
    def scrape_create_ai_list_split(driver, days:list=[], day:datetime=None, now_race_id:str='', now_race_ids:list=[]) -> None:
        """ AIリスト作成
            Args:
                driver (WebDriver): Selenium WebDriver
                days (list): 取得日リスト
                day (datetime): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (list): 現在レースIDリスト
            Returns:
                None
        """
        race_lst = []
        dic = {}
        dic['芝'] = {}
        dic['ダート'] = {}
        for date in days:
            url = 'https://race.netkeiba.com/top/race_list.html?kaisai_date=' + date.strftime('%Y%m%d')
            for i in range(3):
                #タイムアウトのハンドリング
                try:
                    driver.get(url)
                    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'RaceList_DataList')))
                    time.sleep(1)
                except TimeoutException:
                    continue
                else:
                    # 失敗しなかった場合は、ループを抜ける
                    break
            else:
                msg = f'取得失敗(最大実行回数オーバー):{url}'
                raise Exception(msg)
            
            time.sleep(1)

            # データリスト取得
            elements = driver.find_elements(by=By.CLASS_NAME, value='RaceList_DataList')

            # 取得したリストを解析
            for element in elements:
                # 会場取得
                print('会場取得')
                titles = element.find_elements(by=By.CLASS_NAME, value='RaceList_DataTitle')
                place = ''
                for title in titles:
                    pattern = r'\d'
                    parts = title.text.split()
                    for part in parts:
                        if re.search(pattern, part):
                            print(f"{part}")
                        else:
                            print(f"{part}")
                            
                            turf = dic['芝'].get(part)
                            if turf is None:
                                dic['芝'][part] = []
                            dirt = dic['ダート'].get(part)
                            if dirt is None:
                                dic['ダート'][part] = []
                            place = part

                print()
                print('レース情報取得')
                race_data = element.find_elements(by=By.CLASS_NAME, value='RaceList_DataItem')
                for race in tqdm(race_data):
                    # タイトル取得
                    title = race.find_element(by=By.CLASS_NAME, value='RaceList_ItemContent')
                    #print(f'1. {title.text}')
                    if '障' in title.text:
                        print('continue')
                        continue
                        
                    # データ取得
                    data = race.find_element(by=By.CLASS_NAME, value='RaceList_ItemLong')
                    # 距離取得
                    distance = 0
                    #print(f'2. {data.text}')
                    if 'm' in data.text:
                        distance = int(re.findall(r'\d+', data.text)[0])
                        #print(f'3. {distance}')
                    else:
                        raise Exception('距離データが見つかりません') 
                    
                    key = ''
                    if '芝' in data.text:
                        key = '芝'
                        #if distance <= 1200:
                        #    distance=1200
                        #elif 2200 <= distance:
                        #    distance=2200
                    elif 'ダ' in data.text:
                        key = 'ダート'
                        #if distance <= 1200:
                        #    distance=1200
                        #elif 1800 <= distance:
                        #    distance=1800
                    else:
                        print('障害レース')
                        continue
                    distance = Dataset.convert_distance(key, place, distance)
                    count = dic[key][place].count(distance)
                    if count > 0:
                        continue
                        
                    dic[key][place].append(distance)
                    dic[key][place].sort()

        TodayData.exec_dict = dic

    @staticmethod
    def create_execute_time(x, today:datetime) -> datetime:
        """ 実行時間作成
            Args:
                x (str): 実行時間
                today (datetime): 今日の日付
            Returns:
                datetime: 実行時間
        """
        split = str(x).split(':')
        h = int(split[0])
        m = int(split[1])
        #today = dt.datetime.today()
        execute = datetime(today.year, today.month, today.day, h, m)
        correction = -420
        execute = execute + timedelta(seconds=correction)
        return execute

    @staticmethod
    def get_weekday(day_index: int) -> str:
        """ 曜日取得
            Args:
                day_index (int): 曜日インデックス (0:月曜日, 1:火曜日, ..., 6:日曜日)
            Returns:
                str: 曜日
        """
        w_list = ['月', '火', '水', '木', '金', '土', '日']
        return w_list[day_index]

    @staticmethod
    def scrape_win5_race_ids(driver, days:list=[], day:datetime=None, now_race_id:str='', now_race_ids:list=[]) -> None:
        """ WIN5レースID取得
            Args:
                driver (WebDriver): Selenium WebDriver
                days (list): 取得日リスト
                day (datetime): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (list): 現在レースIDリスト
            Returns:
                None
        """
        # レース > 今週のWIN5
        url = 'https://race.netkeiba.com/top/win5.html'

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

        TodayData.win5_list = []
        # 『対象レース』テーブルのレース行指定+各種列取得
        for i in range(2,7):
            # XPath作成
            path = '//*[@id="Netkeiba_Race_Win5"]/div[1]/div/div[1]/div[1]/div/table/tbody/tr[2]/td[{0}]/a'.format(i)
            # Link取得
            link = driver.find_element(By.XPATH, path).get_attribute("href")
            # レースID取得
            race_id = re.findall(r'\d+', link)[0]
            # リストに追加
            TodayData.win5_list.append([race_id, link])

    @staticmethod
    def scrape_time_table(driver, days:list=[], day:datetime=None, now_race_id:str='', now_race_ids:list=[]) -> None:
        """ タイムテーブル取得
            Args:
                driver (WebDriver): Selenium WebDriver
                days (list): 取得日リスト
                day (datetime): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (list): 現在レースIDリスト
            Returns:
                None
        """
        time_table = pd.DataFrame()
        TodayData.time_table = pd.DataFrame()
        TodayData.exists_win5 = False
        
        while len(TodayData.time_table) == 0:
            url = 'https://race.netkeiba.com/top/race_list.html?kaisai_date=' + day.strftime('%Y%m%d')
            for i in range(3):
                #タイムアウトのハンドリング
                try:
                    driver.get(url)
                    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'RaceList_DataList')))
                    time.sleep(1)
                except TimeoutException:
                    continue
                else:
                    # 失敗しなかった場合は、ループを抜ける
                    break
            else:
                msg = f'取得失敗(最大実行回数オーバー):{url}'
                raise Exception(msg)

            # データリスト取得
            elements = driver.find_elements(by=By.CLASS_NAME, value='RaceList_DataList')

            race_id_lst = []
            time_list = []
            # 取得したリストを解析
            for element in elements:
                # レースID
                data_items = element.find_elements(by=By.CLASS_NAME, value='RaceList_DataItem')
                for item in data_items:
                    a = item.find_element(by=By.TAG_NAME, value="a").get_attribute("href")
                    race_id_lst.append(re.findall(r"\d+", a)[0])
                # 出走時間
                tbl_time = element.find_elements(by=By.CLASS_NAME, value='RaceList_Itemtime')
                for item in tbl_time:
                    time_list.append(item.text)
                    
                # win5
                win5 = element.find_elements(by=By.XPATH, value='//*[contains(@class,"Icon_GradeType")]');
                for item in win5:
                    txt = item.get_attribute('class')
                    arr = txt.split(' ')
                    if len(arr) == 2:
                        if arr[1] == 'Icon_GradeType13':
                            TodayData.exists_win5 = True

            add_time_table = pd.DataFrame(data=time_list, index=race_id_lst, columns=['start_time'])
            add_time_table['日付'] = day
            time_table = pd.concat([time_table, add_time_table])

            time_table = time_table.sort_values('start_time', ascending=True)
            # 実行時間作成
            time_table['execute_time'] = time_table['start_time'].map(lambda x: TodayData.create_execute_time(x, day))
            # 実行済み列生成
            time_table['is_executed'] = False
            # win5設定
            time_table['win5'] = False
            
            if TodayData.exists_win5 == True:
                # win5 のレースID取得
                TodayData.scrape_win5_race_ids(driver, days, day)

                cnt=1
                for target in TodayData.win5_list:
                    # 文字数確認
                    race_id = target[0]
                    if len(race_id) != 12:
                        raise Exception(f'{cnt}レース目のIDが正しく取得できていません。 {race_id}')
                    # 対象レースにフラグ設定
                    time_table.loc[race_id,'win5'] = True
                    cnt += 1

                # 時間差を算出
                time_table.loc[:,'diff'] = time_table['execute_time'] - time_table['execute_time'].shift(1)
                # 昼休憩(26分以上)間隔があいている箇所を探す
                insert_row = time_table[time_table['diff'] >= timedelta(seconds=1560)]
                #if len(insert_row) != 1:
                #    raise Exception('win5 実行が複数見つかりました')
                # 行番号を取得
                print(insert_row.index[0])
                row_no = time_table.index.tolist().index(insert_row.index[0])
                # 新規行を作成
                new_row = time_table.iloc[:row_no].tail(1).copy()
                # 実行時間を追加
                new_row['execute_time'] = new_row['execute_time'] + timedelta(seconds=360)
                # インデックス修正
                new_row.index = ['win5']
                # 結合
                time_table = pd.concat([time_table.iloc[:row_no], new_row, time_table.iloc[row_no:]])

            TodayData.time_table = time_table.copy()

    @staticmethod
    def scrape_race_table(driver, days:list=[], day:datetime=None, now_race_id:str='', now_race_ids:list=[]) -> None:
        """ 出馬表取得
            Args:
                driver (WebDriver): Selenium WebDriver
                days (list): 取得日リスト
                day (datetime): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (list): 現在レースIDリスト
            Returns:
                None
        """
        # 出馬表取得
        TodayData.race_df = None
        race_df = pd.DataFrame()
        try:
            print('出馬表取得')
            df = None
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + now_race_id
            # 最大10回実行
            for _ in range(10):
                try:
                    # 失敗しそうな処理
                    print(f'ページ情報取得 {url}')
                    driver.set_page_load_timeout(10)
                    driver.get(url)
                    print('待機')
                    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'RaceTableArea')))
                    time.sleep(1)
                except Exception as e:
                    print(f'except continue {e}')
                    continue
                else:
                    # 失敗しなかった場合は、ループを抜ける
                    break
            else:
                msg = f'出馬表取得失敗:{url}'
                raise Exception(msg)

            # 出馬表取得
            tableElem = driver.find_element(by=By.CLASS_NAME, value='RaceTableArea')
            # 行取得
            trs = tableElem.find_elements(By.TAG_NAME, "tr")

            # ヘッダー解析
            # {key:col_index, value:col_name}
            dic_header = {}
            #　1行目は列情報
            for i in range(0,1):
                # 1行目のth取得
                ths = trs[i].find_elements(By.TAG_NAME, "th")
                # 列分ループ
                for col_index in range(0,len(ths)):
                    # 改行含む文字列をまとめて取得
                    txt = ths[col_index].text.split()
                    if len(txt) != 1:
                        # 要素が複数の場合結合
                        txt = ''.join(txt)
                    else:
                        txt = txt[0]
                    dic_header[col_index] = txt

            # カラムヘッダーは2行なので 2 スタート
            for i in range(2,len(trs)):
                tds = trs[i].find_elements(By.TAG_NAME, "td")
                line = {}
                # ヘッダー基準でデータ取得
                for col_index, col_name in dic_header.items():

                    if '枠' in col_name:
                        line['枠番'] = tds[col_index].text
                    elif 'オッズ' in col_name:
                        line['単勝'] = tds[col_index].text
                    elif '馬体重' in col_name:
                        line['馬体重'] = tds[col_index].text
                    else:
                        line[col_name] = tds[col_index].text

                    if col_name == '馬名':
                        href = tds[col_index].find_element(by=By.TAG_NAME, value='a').get_attribute('href')
                        line['馬名_ID'] = re.findall(r'\d+', href)[0]
                    elif col_name == '騎手':
                        href = tds[col_index].find_element(by=By.TAG_NAME, value='a').get_attribute('href')
                        line['騎手_ID'] = re.findall(r'\d+', href)[0]
                    elif col_name == '厩舎':
                        href = tds[col_index].find_element(by=By.TAG_NAME, value='a').get_attribute('href')
                        title = tds[col_index].find_element(by=By.TAG_NAME, value='a').get_attribute('title')
                        line['所属'] = tds[col_index].text.replace(title, '')
                        line['調教師'] = title
                        line['調教師_ID'] = re.findall(r'\d+', href)[0]

                # 初回判定
                if type(df) != pd.core.frame.DataFrame:
                    df = pd.DataFrame(data=line, columns=list(line.keys()), index=[now_race_id])
                else:
                    row = pd.DataFrame([line])
                    df = pd.concat([df, row])
            # date
            df['日付'] = [day] * len(df)
            df['日付'] = pd.to_datetime(df['日付'], format='%Y-%m-%d')

            # レース名
            race_name = driver.find_element(by=By.CLASS_NAME, value='RaceName')
            df['レース名'] = [race_name.text] * len(df)

            # グレード
            grades = race_name.find_elements(by=By.XPATH, value='//*[contains(@class,"Icon_GradeType")]');
            grade = 'なし'
            for item in grades:
                txt = item.get_attribute('class')
                arr = txt.split(' ')
                if len(arr) == 2:
                    if arr[1] == 'Icon_GradeType1':
                        grade = 'G1'
                    elif arr[1] == 'Icon_GradeType2':
                        grade = 'G2'
                    elif arr[1] == 'Icon_GradeType3':
                        grade = 'G3'
                    else:
                        grade = 'なし'
            df['レース名1'] = [grade] * len(df)

            race_data_1 = driver.find_elements(by=By.CLASS_NAME, value='RaceData01')

            df['レースタイプ'] = [np.nan] * len(df)
            df['レース周り'] = [np.nan] * len(df)
            #df['outside'] = [np.nan] * len(df)
            df['距離'] = [np.nan] * len(df)
            df['馬場'] = [np.nan] * len(df)
            df['天気'] = [np.nan] * len(df)
            for info in race_data_1:
                texts = re.findall(r'\w+', info.text)
                for text in texts:
                    if len(df[df['レースタイプ'].notnull()]) == 0:
                        if '芝' in text:
                            df['レースタイプ'] = '芝'
                        elif 'ダ' in text:
                            df['レースタイプ'] = 'ダート'
                        elif '障' in text:
                            df['レースタイプ'] = '障害'
                    if '右' in text:
                        df['レース周り'] = '右'
                    if '左' in text:
                        df['レース周り'] = '左'
                    if '直線' in text:
                        df['レース周り'] = '直線'                       
                    #if '外' in text:
                    #    df['outside'] = '外'
                    if 'm' in text:
                        df['距離'] = re.findall(r'\d+', text) * len(df)
                    if text in ['良', '重']:
                        df['馬場'] = text
                    if '稍' in text:
                        df['馬場'] = '稍重'
                    if '不' in text:
                        df['馬場'] = '不良'
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df['天気'] = text

            race_data_2 = driver.find_elements(by=By.CLASS_NAME, value='RaceData02')

            df['詳細条件_0'] = [''] * len(df)
            df['詳細条件_1'] = [''] * len(df)
            df['詳細条件_2'] = [''] * len(df)
            df['詳細条件_3'] = [''] * len(df)
            df['詳細条件_4'] = [''] * len(df)

            for info in race_data_2:
                texts = re.findall(r'\w+', info.text)
                num = len(texts)
                plc = ''
                cnd = ''
                for t in range(0, num):
                    ex = re.findall(r'\D+', texts[t])[0]
                    if ex == '頭':
                        break
                    if t == 0:
                        df['開催数'] = [texts[t]] * len(df)
                        plc += texts[t]
                    elif t == 1:
                        df['会場'] = [texts[t]] * len(df)
                        plc += texts[t]
                    elif t == 2:
                        df['開催日'] = [texts[t]] * len(df)
                        plc += texts[t]
                        df['held'] = [plc] * len(df)
                    elif t == 3:
                        cnd = str(texts[t]).replace('サラ系', '')
                    elif t == 4:
                        cnd += texts[t]
                        df['詳細条件_0'] = [cnd]* len(df)
                    elif t == 5:
                        df['詳細条件_1'] = [texts[t]] * len(df)
                    elif t == 6:
                        df['詳細条件_2'] = [texts[t]] * len(df)
                    elif t == 7:
                        df['詳細条件_3'] = [texts[t]] * len(df)
                    elif t == 8:
                        df['詳細条件_4'] = [texts[t]] * len(df)
                    else:
                        break        
            
            # インデック修正
            df.index = [now_race_id] * len(df)
            df['頭数'] = len(df)
            df['race_id'] = df.index
            
            df = df[(df['印']!='除外')&(df['印']!='取消')]

            race_df = df
        except Exception as e:
            msg = f'出馬表取得失敗:{url}\r\n{e}'          
            raise Exception(msg)
        
        # 各馬情報
        horse_ids = race_df['馬名_ID'].unique()
        TodayData.peds_df = None
        peds_df = pd.DataFrame()
        try:
            print('各馬情報取得')
            # ホースIDループ
            for horse_id in tqdm(horse_ids):
                try:
                    url = 'https://db.netkeiba.com/horse/' + horse_id
                    # 最大5回実行
                    for _ in range(5): 
                        try:
                            driver.set_page_load_timeout(10)
                            driver.get(url)
                            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'db_prof_table')))
                            time.sleep(1)
                        except Exception as e:
                            print(f'各馬情報取得 continue {horse_id}')
                            continue
                        else:
                            # 失敗しなかった場合は、ループを抜ける
                            break
                    else:
                        msg = f'owner_id 取得失敗(最大実行回数オーバー):{url}'
                        raise Exception(msg)

                    tbody = driver.find_element(by=By.CLASS_NAME, value='db_prof_table')
                    trs = tbody.find_elements(by=By.TAG_NAME, value='tr')
                    for tr in trs:
                        if '馬主' not in tr.text:
                            continue
                        td = tr.find_element(by=By.TAG_NAME, value='td')
                        #print(td.text)
                        href = td.find_element(by=By.TAG_NAME, value='a').get_attribute('href')
                        owner_id = re.findall(r'\d+', href)[0]
                        #print(owner_id)
                        race_df.loc[race_df['馬名_ID']==horse_id, '馬主'] = td.text
                        race_df.loc[race_df['馬名_ID']==horse_id, '馬主_ID'] = owner_id
                #存在しないrace_idを飛ばす
                except IndexError:
                    print(f'IndexError {horse_id}')
                    Notification.send(f'TodayData scrape_race_table IndexError {horse_id}: {e}')
                    continue
                #wifiの接続が切れた時などでも途中までのデータを返せるようにする
                except Exception as e:
                    print(f'Exception {horse_id} {e}')
                    Notification.send(f'TodayData scrape_race_table Exception {horse_id}: {e}')
                    continue
                #Jupyterで停止ボタンを押した時の対処    
                except:
                    print(f'Exception {horse_id}')
                    break
                
                data = None
                try:
                    # 接続先
                    url = f'https://db.netkeiba.com/horse/ped/{horse_id}/'
                    for _ in range(3):
                        try:
                            driver.set_page_load_timeout(10)
                            driver.get(url)
                            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'blood_table')))
                            time.sleep(1)
                        except Exception as e:
                            print('pedigree table continue')
                            continue
                        else:
                            # 失敗しなかった場合は、ループを抜ける
                            break
                    else:
                        msg = f'取得失敗:{url}'
                        driver.close()
                        driver.quit()
                        raise TimeoutException(msg)

                    data = BeautifulSoup(driver.page_source, 'html.parser')
                #存在しないrace_idを飛ばす
                except IndexError:
                    print(f'IndexError {horse_id}')
                    Notification.send(f'TodayData scrape_race_table IndexError {horse_id}: {e}')
                    continue
                #wifiの接続が切れた時などでも途中までのデータを返せるようにする
                except Exception as e:
                    print(f'Exception {horse_id} {e}')
                    Notification.send(f'TodayData scrape_race_table Exception {horse_id}: {e}')
                    continue
                #Jupyterで停止ボタンを押した時の対処    
                except:
                    print(f'Exception {horse_id}')
                    break

                # テーブル取得
                table = data.find(class_='blood_table')
                # td取得
                tds = table.find_all('td')
                if len(tds) != 62:
                    print(f'len != 62, {horse_id}')

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
                            print(f'{horse_id} is none peds_type')

                    if type(name) != type(None):
                        dic_ped[f'peds_{cnt:02}'] = [name.get_text(strip=True)]
                    else:
                        dic_ped[f'peds_{cnt:02}'] = [np.nan]
                        print(f'{horse_id} peds_{cnt:02} is none name')

                    if len(ids) > 3: 
                        dic_ped[f'peds_{cnt:02}_ID'] = [ids[3]]
                    else:
                        dic_ped[f'peds_{cnt:02}_ID'] = [np.nan]
                        print(f'{horse_id} peds_{cnt:02} is none id')

                    # カウンタ加算
                    cnt += 1

                # データフレーム変換
                df = pd.DataFrame.from_dict(dic_ped)
                df.index = [horse_id] * len(df)
                # 馬名_ID 追加
                df['馬名_ID'] = horse_id

                peds_df = pd.concat([peds_df, df])
        except Exception as e:
            msg = f'血統表取得失敗:{url}\r\n{e}'          
            raise Exception(msg)

        # 調教
        TodayData.training_df = None
        try:
            print('調教情報取得')
            training_df = None
            dic_header = {}
            for _ in range(5):
                try:
                    url = f'https://race.netkeiba.com/race/oikiri.html?race_id={now_race_id}'
                    for _ in range(3):
                        try:
                            driver.set_page_load_timeout(10)
                            driver.get(url)
                            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'race_table_01')))
                            time.sleep(1)
                        except Exception as e:
                            print(f'{now_race_id} race_table_01 continue')
                            continue
                        else:
                            # 失敗しなかった場合は、ループを抜ける
                            break
                    else:
                        msg = f'取得失敗:{url}'
                        driver.close()
                        driver.quit()
                        raise TimeoutException(msg)
                    data = BeautifulSoup(driver.page_source, 'html.parser')
                    # tr 取得
                    trs = data.find_all('tr')
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
                    df_add.index = [now_race_id] * len(df_add)
                    df_add['race_id'] = df_add.index

                    training_df = df_add
                    break
                #存在しないrace_idを飛ばす
                except IndexError:
                    print(f'IndexError {now_race_id}')
                    Notification.send(f'TodayData scrape_race_table IndexError {now_race_id}: {e}')
                    continue
                #wifiの接続が切れた時などでも途中までのデータを返せるようにする
                except Exception as e:
                    print(f'Exception {now_race_id} {e}')
                    Notification.send(f'TodayData scrape_race_table Exception {now_race_id}: {e}')
                    continue
                #Jupyterで停止ボタンを押した時の対処    
                except:
                    print(f'Exception {now_race_id}')
                    break
        except Exception as e:
            msg = f'調教データ取得失敗:{url}\r\n{e}'
            raise Exception(msg)

        print(day)
        str_day = datetime.strftime(day, '%Y%m%d')
        dir_path = f'./html/predict/{str_day}/{now_race_id}'
        if os.path.isdir(dir_path) == False:
            print(f'create is {dir_path} folder')
            os.makedirs(dir_path)
            
        TodayData.race_df = race_df
        TodayData.peds_df = peds_df
        TodayData.training_df = training_df
        
        race_df.to_pickle(f'{dir_path}/race_table.pickle')
        peds_df.to_pickle(f'{dir_path}/peds_table.pickle')
        training_df.to_pickle(f'{dir_path}/training_table.pickle')

    @staticmethod
    def scrape_refund_table(driver, days:list=[], day:datetime=None, now_race_id:str='', now_race_ids:list=[]) -> None:
        """ 払戻金取得
            Args:
                driver (WebDriver): Selenium WebDriver
                days (list): 取得日リスト
                day (datetime): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (list): 現在レースIDリスト
            Returns:
                None
        """
        cols = [0, 1, 2, 3]
        result_df = pd.DataFrame(index=[], columns=cols)
        for race_id in now_race_ids:
            print(race_id)
            url = 'https://race.netkeiba.com/race/result.html?race_id=' + race_id
            for _ in range(0,3):
                try:
                    print(url)
                    driver.set_page_load_timeout(10)
                    driver.get(url)
                    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'Payout_Detail_Table')))
                    time.sleep(1)

                    tbodys = driver.find_elements(by=By.CLASS_NAME, value='Payout_Detail_Table')

                    df = pd.DataFrame(index=[], columns=cols)
                    for tbody in tbodys:
                        trs = tbody.find_elements(by=By.TAG_NAME, value='tr')
                        for tr in trs:
                            lst = []
                            tr_class = tr.get_attribute("class")
                            th = tr.find_element(by=By.TAG_NAME, value='th')
                            lst.append(th.text)
                            tds = tr.find_elements(by=By.TAG_NAME, value='td')
                            for td in tds:
                                td_class = td.get_attribute("class")
                                if td_class == 'Result':
                                    lst.append(td.text)
                                elif td_class == 'Payout':
                                    lst.append(td.text)
                                elif td_class == 'Ninki':
                                    lst.append(td.text)
                            df = pd.concat([df, pd.DataFrame(data=lst).T], axis=0)
                    df.index = [race_id] * len(df)

                    result_df = pd.concat([result_df, df])
                except Exception as e:
                    print(race_id, e)
                    time.sleep(3)
                    continue
                else:
                    break

        TodayData.result_df = result_df
