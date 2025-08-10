import os
import re
import pandas as pd
import polars as pl
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup


class RefundTable:
    """ 払い戻しテーブルクラス """
    def __init__(self, refund: pl.DataFrame, is_today: bool=False):
        """ コンストラクタ 
            Args:
                refund (DataFrame): 払い戻しテーブル
                is_today (bool): 当日データかどうか
        """
        print('払い戻し 変換')
        self.cnv_cols=["着順", "オッズ", "人気"]
        self.cnv_dic={
                '着順': 'win',
                'オッズ': 'ref',
                '人気': 'pop'
            }
        if is_today == True:
            print('当日用データ変換')
            #refund[1] = refund[1].map(lambda x: x.replace('\n', 'br'))
            refund = refund.with_columns(pl.col("1").str.replace_all("\n", "br").alias("1"))
            #refund[2] = refund[2].map(lambda x: x.replace('\n', 'br'))
            refund = refund.with_columns(pl.col("2").str.replace_all("\n", "br").alias("2"))
            #refund[2] = refund[2].map(lambda x: x.replace('円', ''))
            refund = refund.with_columns(pl.col("2").str.replace_all("円", "").alias("2"))
            #refund[2] = refund[2].map(lambda x: x.replace(',', ''))
            refund = refund.with_columns(pl.col("2").str.replace_all(",", "").alias("2"))
            #refund[3] = refund[3].map(lambda x: x.replace('\n', 'br'))
            refund = refund.with_columns(pl.col("3").str.replace_all("\n", "br").alias("3"))
            #refund[3] = refund[3].map(lambda x: x.replace('人気', ''))
            refund = refund.with_columns(pl.col("3").str.replace_all("人気", "").alias("3"))
        
        # 元テーブル保存
        refund = refund.rename({
                '0': '券種',
                '1': '着順',
                '2': 'オッズ',
                '3': '人気'
            })
        self._refund_table = refund
        self._place = self.cnv_table(refund, "複勝")
        self._win = self.cnv_table(refund, "単勝")
        self._wide = self.cnv_table(refund, "ワイド")
        self._bracket_quinella = self.cnv_table(refund, "枠連")
        self._quinella = self.cnv_table(refund, "馬連")
        self._exacta = self.cnv_table(refund, "馬単")
        self._trio = self.cnv_table(refund, "3連複")
        self._tierce = self.cnv_table(refund, "3連単")

    def replace(self, x) -> str:
        """ 置換処理
            Args:
                x (str): 置換対象文字列
            Returns:
                str: 置換後文字列
        """
        s = str(x)
        if ',' in s:
            return s.replace(',','')
        else:
            return s

    def convert(self, x: str) -> set:
        """ 変換処理
            Args:
                x (str): 変換対象文字列
            Returns:
                set: 変換後セット
        """
        l_si = re.findall(r'\d+', x)
        l_si_i = [int(s) for s in l_si]
        return set(l_si_i)

    def check(self, s: str) -> str:
        x=s
        if ',' in x:
            x=x.replace(',','')
        if ' - ' in x:
            x=x.replace(' - ', ' ')
        if ' → ' in x:
            x=x.replace(' → ', ' ')
        if "br" in x:
            return x
        return f"{x} br"

    def title(self, s: str) -> str:
        if (s == "三連複") | (s == "3連複"):
            return "3連複"
        elif (s == "三連単") | (s == "3連単"):
            return "3連単"
        return s

    # テーブル
    def cnv_table(self, table: pl.DataFrame, target: str) -> pd.DataFrame:
        """ 払戻表 変換
            Args:
                table (DataFrame): 払戻表
            Returns:
                DataFrame: テーブル
        """
        print(f'払戻表 {target} 変換')
        df = table.with_columns(
                pl.col("券種").map_elements(lambda x: self.title(x), return_dtype=pl.Utf8).alias("券種")
                )
        target_cols = self.cnv_cols.copy()
        target_cols.append('race_id')
        df = df.filter(pl.col("券種") == target).select(target_cols)
        for col in self.cnv_cols:
            title = self.cnv_dic[col]
            df = df.with_columns(
                # `br` で区切ってリスト化 → `Utf8` にキャスト
                pl.col(col)
                .map_elements(lambda x: self.check(x), return_dtype=pl.Utf8)
                .str.split("br")
                .list.eval(pl.element().cast(pl.Utf8))
                .list.to_struct(fields=[f"{title}_{i}" for i in range(df[col].str.split("br").list.len().max())])
                .alias(f"{title}_struct")
            ).unnest(f"{title}_struct")  # Struct をカラムに展開

        df = df.drop(self.cnv_cols)

        # 全ての行が null のカラムを削除
        df = df.select([
            col for col in df.columns 
            if not df[col].drop_nulls().is_empty() and not (df[col].drop_nulls() == "").all()
        ])
        df = df.with_columns([
            pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
            for col in df.columns
            if df[col].dtype == pl.Utf8  # 文字列列のみ対象
        ])
        df = df.to_pandas()
        df.index = df["race_id"]

        return df

    @property
    def refund_table(self):
        return self._refund_table
    @property
    def place(self):
        return self._place
    @property
    def win(self):
        return self._win
    @property
    def wide(self):
        return self._wide
    @property
    def bracket_quinella(self):
        return self._bracket_quinella
    @property
    def quinella(self):
        return self._quinella
    @property
    def exacta(self):
        return self._exacta
    @property
    def trio(self):
        return self._trio
    @property
    def tierce(self):
        return self._tierce

    @staticmethod
    def create_data(start: int, end: int) -> None:
        """ データ作成
            Args:
                start (int): 開始年
                end (int): 終了年
            Returns:
                None
        """
        for year in range(start, end):
            # ファイル取得
            files = os.listdir(f'./html/race_result_db/{year}/')
            df = pd.DataFrame()
            is_error = False
            for file in tqdm(files):
                try:
                    if '.bin' not in file:
                        print(f'Not target {file}')
                        continue

                    html = None
                    with open(f'./html/race_result_db/{year}/{file}') as f:
                        html = f.read()

                    html = html.replace('<br/>', 'br')
                    #dfs = pd.read_html(html)
                    dfs = pd.read_html(StringIO(html))

                    #dfsの1番目に単勝-馬連、2番目にワイド-三連単がある
                    df_add = pd.concat([dfs[1], dfs[2]])
                    # index 設定
                    df_add.index = [file.replace('.bin', '')] * len(df_add)

                    if len(df) == 0:
                        df = df_add
                    else:
                        df = pd.concat([df,df_add])
                except Exception as e:
                    print(file, e)
                    
                    html = None
                    with open(f'./html/race_result_db/{year}/{file}') as f:
                        html = f.read()
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    tables = soup.find_all(class_='pay_table_01')
                    read_df = pd.DataFrame()

                    for table in tables:
                        dic = {}
                        trs = table.find_all('tr')
                        for tr in trs:
                            ths = tr.find_all('th')
                            for th in ths:
                                dic[0] = [th.get_text(strip=True)]
                            tds = tr.find_all('td')
                            cnt=1
                            for td in tds:
                                dic[cnt] = [td.text.replace(' ', '').replace('\n\n', ' br ').replace('\n', '').replace('→', ' → ').replace('-', ' - ')]
                                cnt += 1
                            read_df = pd.concat([read_df, pd.DataFrame(data=dic)])
                    read_df.index = [file.split('.')[0]] * len(read_df)
                    
                    if len(df) == 0:
                        df = read_df
                    else:
                        df = pd.concat([df,read_df])                    
                    continue

            df.to_pickle(f'./html/refund_table/{year}.pickle')

    @staticmethod
    def create_max_value(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """ 最大値取得
            Args:
                df (DataFrame): データフレーム
                col_name (str): カラム名
            Returns:
                DataFrame: 最大値データフレーム
        """
        # 'pop_' で始まるカラムを取得
        target_columns = [col for col in df.columns if col.startswith("pop_")]

        # 各カラムを float 型に変換して最大値を取得
        max_df = df.select([
            pl.max_horizontal(*target_columns).alias(f"{col_name}人気"),
            pl.col("race_id")
        ])

        return max_df

