import time
from datetime import datetime
from typing import Callable, List, Optional

from selenium.webdriver import Edge
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from src.utils.notification import Notification
from src.executors.user_info import UserInfo
from src.utils.json_serializer import JsonSerializer

class NetKeibaAccesser:
    """ ネット競馬アクセス """

    @staticmethod
    def run(func: Callable, days: List[datetime] = [], day: datetime = None, now_race_id: str = '', now_race_ids: List[str] = []):
        """ ネット競馬アクセス 
            Args:
                func (Callable): 実行関数
                days (List[datetime]): 取得日リスト
                day (Optional[datetime]): 取得日
                now_race_id (str): 現在レースID
                now_race_ids (List[str]): 現在レースIDリスト
            Returns:
                None
        """

        user = JsonSerializer.read('./html/data/maxi_info.json')
        user_info = UserInfo.from_dict(user)

        #options = ChromeOptions()
        options = EdgeOptions()
        options.add_argument("--disable-javascript")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_experimental_option('extensionLoadTimeout', 10000)
        #options.add_argument('--headless')

        # Selenium ver3.xを使用している場合
        #driver = Chrome(executable_path=ChromeDriverManager().install(), options=options)

        # Selenium ver4.xを使用している場合
        #chrome_service = ChromeService.Service(executable_path=ChromeDriverManager().install())
        #driver = Chrome(service=chrome_service, options=options)
        
        driver = Edge(options=options)

        # タイムアウト設定
        driver.set_page_load_timeout(10)
        try:
            # url設定
            url='https://regist.netkeiba.com/account/?pid=login'

            for i in range(3):
                #タイムアウトのハンドリング
                try:
                    driver.get(url)
                except TimeoutException:
                    continue
                else:
                    # 失敗しなかった場合は、ループを抜ける
                    break
            else:
                msg = f'取得失敗(最大実行回数オーバー):{url}'
                raise Exception(msg)

            # ログインID 検索
            login_id = driver.find_element(by=By.XPATH, value='//*[@id="contents"]/div/form/div/ul/li[1]/input')
            time.sleep(1)
            # ログインID 入力
            login_id.send_keys(user_info.net_keiba_id)
            time.sleep(1)

            # パスワード 検索
            password = driver.find_element(by=By.XPATH, value='//*[@id="contents"]/div/form/div/ul/li[2]/input')
            time.sleep(1)
            # パスワード 入力
            password.send_keys(user_info.net_keiba_password)
            time.sleep(1)

            # ログインボタン検索
            login = driver.find_element(by=By.XPATH, value='//*[@id="contents"]/div/form/div/div[1]/input')
            # ログインボタンクリック
            login.click()
            time.sleep(1)

            # 処理呼び出し
            func(driver, days, day, now_race_id, now_race_ids)
        except Exception as e:
            print(e)
            Notification.send(e)
            raise
        finally:
            # driver閉じる
            driver.close()
            # driver終了
            driver.quit()
