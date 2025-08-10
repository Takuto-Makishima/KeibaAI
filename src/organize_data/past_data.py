import os
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from src.utils.path_manager import PathManager
from src.table_data.race_table import RaceTable

class PastData:
    """ 過去データを作成するクラス """
    @staticmethod
    def get_date_list(date_list: list) -> list:
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
    def cnv_class(before_class: str, today_class: str) -> str:
        lst = ['新馬クラス', '未勝利クラス', '1勝クラス', '2勝クラス', 
               '3勝クラス', 'オープンクラス', 'G3クラス', 'G2クラス', 'G1クラス']
        try:
            before_idx = lst.index(before_class)
            today_idx = lst.index(today_class)
        except ValueError:
            return "不明"  # クラス名がリストにない場合

        if today_idx > before_idx:
            return "昇級"
        elif today_idx < before_idx:
            return "降級"
        else:
            return "据え置き"

    @staticmethod
    def get_average_score(today_df: pl.DataFrame, past_df: pl.DataFrame, num: int) -> pl.DataFrame:
        """ 指定されたカラムの平均値を取得する関数
            Args:
                today_df: 今日のデータフレーム
                past_df: 過去のデータフレーム
                num: レース数
            Returns:
                pl.DataFrame: 平均値を持つデータフレーム
        """
        ave = past_df.group_by("馬名_ID").agg(
            pl.col("order").mean().alias(f"着順_{num}R"),
            pl.col("ﾀｲﾑ指数").mean().alias(f"ﾀｲﾑ指数_{num}R"),
            pl.col("1角").mean().alias(f"1角_{num}R"),
            pl.col("4角").mean().alias(f"4角_{num}R"),
            pl.col("位置取り").mean().alias(f"位置取り_{num}R"),
            pl.col("上り").mean().alias(f"上り_{num}R"),
            pl.col("単勝").mean().alias(f"単勝_{num}R"),
            pl.col("人気").mean().alias(f"人気_{num}R"),
            pl.col("賞金").mean().alias(f"賞金_{num}R"),
            pl.col('出走前馬レート').mean().alias(f'出走前馬レート_{num}R'),
            pl.col('出走前レース内馬レート').mean().alias(f'出走前レース内馬レート_{num}R'),
            pl.col('出走後馬レート').mean().alias(f'出走後馬レート_{num}R'),
            pl.col('出走後レース内馬レート').mean().alias(f'出走後レース内馬レート_{num}R')
        )
        today_df = today_df.join(ave, on="馬名_ID", how="left")
        return today_df

    @staticmethod
    def get_best_score(today_df: pl.DataFrame, past_df: pl.DataFrame, num: str) -> pl.DataFrame:
        """ 指定されたカラムのベストスコアを取得する関数
            Args:
                today_df: 今日のデータフレーム
                past_df: 過去のデータフレーム
                num: レース数
            Returns:
                pl.DataFrame: ベストスコアを持つデータフレーム
        """
        conditions = {
            "完全同一条件": ["馬名_ID", "レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"],
            "同一条件": ["馬名_ID", "レースタイプ", "会場_ID", "距離", "馬場"],
            "会場距離条件": ["馬名_ID", "レースタイプ", "会場_ID", "距離"],
            "馬場条件": ["馬名_ID", "レースタイプ", "距離", "馬場"],
            "年齢条件": ["馬名_ID", "レースタイプ", "距離", "年齢条件"],
            "クラス条件": ["馬名_ID", "レースタイプ", "距離", "クラス"],
            "距離条件": ["馬名_ID", "レースタイプ", "距離"],
        }
        for key, cols in conditions.items():
            # 過去データの集約
            grouped = past_df.group_by(cols).agg(
                pl.col("タイム").min().alias(f"{num}{key}ベストタイム"),
                pl.col("上り").min().alias(f"{num}{key}ベスト上り"),
                pl.col("ﾀｲﾑ指数").max().alias(f"{num}{key}ベストﾀｲﾑ指数"),
                pl.col("出走前馬レート").max().alias(f"{num}{key}ベスト出走前馬レート"),
            )
            today_df = today_df.join(grouped, on=cols, how="left")
        return today_df

    @staticmethod
    def get_jockey_past_data(today_df: pl.DataFrame, past_df: pl.DataFrame, num: str) -> pl.DataFrame:
        """ 指定の順位条件を満たすデータ数を集計し、today_df にマージ
            Args:
                today_df: 今日のデータフレーム
                past_df: 過去のデータフレーム
                num: レース数
            Returns:
                pl.DataFrame: 騎手の過去データを持つデータフレーム
        """
        conditions = {
            "総合": ["馬名_ID", '騎手_ID'],
            "完全同一条件": ["馬名_ID", '騎手_ID', "レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"],
            "同一条件": ["馬名_ID", '騎手_ID', "レースタイプ", "会場_ID", "距離", "馬場"],
            "会場距離条件": ["馬名_ID", '騎手_ID', "レースタイプ", "会場_ID", "距離"],
        }
        total="騎乗数"
        win_count="騎乗勝利数"
        win_rate="騎乗勝率"
        sec_count="騎乗連対数"
        sec_rate="騎乗連対率"
        thr_count="騎乗複勝数"
        thr_rate="騎乗複勝率"
        out_count="騎乗馬券外数"
        out_rate="騎乗馬券外率"
        for key, cols in conditions.items():
            total_name = f"{num}{key}{total}" if key != "総合" else f'{num}騎乗総数'
            win_count_name = f"{num}{key}{win_count}" if key != "総合" else f'{num}{win_count}'
            win_rate_name = f"{num}{key}{win_rate}" if key != "総合" else f'{num}{win_rate}'
            sec_count_name = f"{num}{key}{sec_count}" if key != "総合" else f'{num}{sec_count}'
            sec_rate_name = f"{num}{key}{sec_rate}" if key != "総合" else f'{num}{sec_rate}'
            thr_count_name = f"{num}{key}{thr_count}" if key != "総合" else f'{num}{thr_count}'
            thr_rate_name = f"{num}{key}{thr_rate}" if key != "総合" else f'{num}{thr_rate}'
            out_count_name = f"{num}{key}{out_count}" if key != "総合" else f'{num}{out_count}'
            out_rate_name = f"{num}{key}{out_rate}" if key != "総合" else f'{num}{out_rate}'
            grouped = past_df.group_by(cols).agg(
                pl.len().cast(pl.Float64).alias(total_name),
                pl.col("order").filter(pl.col("order") <= 1).count().cast(pl.Float64).alias(win_count_name),
                pl.col("order").filter(pl.col("order") <= 2).count().cast(pl.Float64).alias(sec_count_name),
                pl.col("order").filter(pl.col("order") <= 3).count().cast(pl.Float64).alias(thr_count_name),
                pl.col("order").filter(pl.col("order") >= 4).count().cast(pl.Float64).alias(out_count_name)
            ).with_columns(
                pl.when(pl.col(total_name) > 0).then(pl.col(win_count_name) / pl.col(total_name)).otherwise(0).alias(win_rate_name),
                pl.when(pl.col(total_name) > 0).then(pl.col(sec_count_name) / pl.col(total_name)).otherwise(0).alias(sec_rate_name),
                pl.when(pl.col(total_name) > 0).then(pl.col(thr_count_name) / pl.col(total_name)).otherwise(0).alias(thr_rate_name),
                pl.when(pl.col(total_name) > 0).then(pl.col(out_count_name) / pl.col(total_name)).otherwise(0).alias(out_rate_name)
            )
            today_df = today_df.join(grouped, on=cols, how="left")
        return today_df

    @staticmethod
    def get_aggregate_past_data(today_df: pl.DataFrame, past_df: pl.DataFrame, num: str) -> pl.DataFrame:
        """ 指定の順位条件を満たすデータ数を集計し、today_df にマージ
            Args:
                today_df: 今日のデータフレーム
                past_df: 過去のデータフレーム
                num: レース数
            Returns:
                pl.DataFrame: 集計結果を持つデータフレーム
        """
        conditions = {
            "総": ["馬名_ID"],
            "完全同一条件": ["馬名_ID", "レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"],
            "同一条件": ["馬名_ID", "レースタイプ", "会場_ID", "距離", "馬場"],
            "会場距離条件": ["馬名_ID", "レースタイプ", "会場_ID", "距離"],
            "レースタイプ": ["馬名_ID", "レースタイプ"],
            "会場": ["馬名_ID", "レースタイプ", "会場_ID"],
            "距離": ["馬名_ID", "レースタイプ", "距離"],
            "馬場": ["馬名_ID", "レースタイプ", "馬場"],
            "年齢条件": ["馬名_ID", "レースタイプ", "年齢条件"],
            "クラス": ["馬名_ID", "レースタイプ", "クラス"],
        }
        total="総数"
        win_count="勝利数"
        win_rate="勝率"
        sec_count="連対数"
        sec_rate="連対率"
        thr_count="複勝数"
        thr_rate="複勝率"
        out_count="馬券外数"
        out_rate="馬券外率"
        for key, cols in conditions.items():
            total_name = f"{num}{key}{total}" if key != "総" else f'{num}総出走数'
            win_count_name = f"{num}{key}{win_count}" if key != "総" else f'{num}{win_count}'
            win_rate_name = f"{num}{key}{win_rate}" if key != "総" else f'{num}{win_rate}'
            sec_count_name = f"{num}{key}{sec_count}" if key != "総" else f'{num}{sec_count}'
            sec_rate_name = f"{num}{key}{sec_rate}" if key != "総" else f'{num}{sec_rate}'
            thr_count_name = f"{num}{key}{thr_count}" if key != "総" else f'{num}{thr_count}'
            thr_rate_name = f"{num}{key}{thr_rate}" if key != "総" else f'{num}{thr_rate}'
            out_count_name = f"{num}{key}{out_count}" if key != "総" else f'{num}{out_count}'
            out_rate_name = f"{num}{key}{out_rate}" if key != "総" else f'{num}{out_rate}'
            grouped = past_df.group_by(cols).agg(
                pl.len().cast(pl.Float64).alias(total_name),
                pl.col("order").filter(pl.col("order") <= 1).count().cast(pl.Float64).alias(win_count_name),
                pl.col("order").filter(pl.col("order") <= 2).count().cast(pl.Float64).alias(sec_count_name),
                pl.col("order").filter(pl.col("order") <= 3).count().cast(pl.Float64).alias(thr_count_name),
                pl.col("order").filter(pl.col("order") >= 4).count().cast(pl.Float64).alias(out_count_name)
            ).with_columns(
                pl.when(pl.col(total_name) > 0).then(pl.col(win_count_name) / pl.col(total_name)).otherwise(0).alias(win_rate_name),
                pl.when(pl.col(total_name) > 0).then(pl.col(sec_count_name) / pl.col(total_name)).otherwise(0).alias(sec_rate_name),
                pl.when(pl.col(total_name) > 0).then(pl.col(thr_count_name) / pl.col(total_name)).otherwise(0).alias(thr_rate_name),
                pl.when(pl.col(total_name) > 0).then(pl.col(out_count_name) / pl.col(total_name)).otherwise(0).alias(out_rate_name)
            )
            today_df = today_df.join(grouped, on=cols, how="left")

            late_start = (
                past_df
                .group_by(cols)
                .agg([
                    pl.col("出遅れ").sum().alias(f"{num}{key}出遅れ回数"),
                    pl.len().alias(f"{num}{key}出走回数")
                ])
                .with_columns(
                    (pl.col(f"{num}{key}出遅れ回数") / pl.col(f"{num}{key}出走回数")).alias(f"{num}{key}出遅れ確率")
                )
            )
            today_df = today_df.join(late_start, on=cols, how="left")
        return today_df

    @staticmethod
    def create_past_data_extra_core(date: pl.Date, day: pl.Date, target_df: pl.DataFrame, result_df: pl.DataFrame) -> pl.DataFrame:
        """ 過去データを作成するコア関数
            Args:
                date: 取得日
                day: 対象日
                target_df: 対象データフレーム
                result_df: 過去データフレーム
            Returns:
                pl.DataFrame: 過去データを持つデータフレーム
        """
        # 該当日のデータ取得
        today_df = target_df.filter(pl.col("日付") == day)
        # 該当日の出走馬ID取得
        horse_ids = today_df["馬名_ID"].unique(maintain_order=True).to_list()
        # 出走馬の出走履歴取得
        past_df = result_df.filter(pl.col("馬名_ID").is_in(horse_ids) & (pl.col("日付") < date)).sort("日付", descending=True)
        # レース間隔計算
        latest_dates = past_df.group_by("馬名_ID").agg(
            pl.col("日付").max().alias("前走日付")
        )
        race_intervals = today_df.select(["馬名_ID", "日付"]).join(
            latest_dates, on="馬名_ID", how="left"
        )
        race_intervals = race_intervals.with_columns(
            pl.when(pl.col("前走日付").is_not_null())
            .then((pl.col("日付") - pl.col("前走日付")).dt.total_days())
            .otherwise(None)
            .cast(pl.Int64)  # Int64 にキャスト
            .alias("レース間隔")
        )
        today_df = today_df.join(race_intervals.select(["馬名_ID", "レース間隔"]), on="馬名_ID", how="left")
        # 脚質
        latest_tactics_with_count = (
            past_df
            .drop_nulls("脚質")  # None や NaN を除外
            .group_by(["馬名_ID", "脚質"])
            .agg([
                pl.col("日付").max().alias("最新日付"),  # 各脚質の最新日付
                pl.len().alias("脚質カウント")  # 各脚質の出現回数
            ])
        )
        final_df = (
            latest_tactics_with_count
            .group_by("馬名_ID")
            .agg([
                pl.col("脚質").filter(pl.col("脚質カウント") == pl.col("脚質カウント").max()).alias("候補脚質"),  # 最頻脚質候補
                pl.col("最新日付").filter(pl.col("脚質カウント") == pl.col("脚質カウント").max()).alias("候補日付")
            ])
            .explode(["候補脚質", "候補日付"])  # 最頻脚質が複数ある場合、行を分割
            .sort(by=["馬名_ID", "候補日付"], descending=True)  # 日付が新しい順にソート
            .group_by("馬名_ID")
            .agg(pl.col("候補脚質").first().alias("平均脚質"))  # 最も新しい日付の脚質を選択
        )
        if "脚質" in today_df.columns:
            today_df = today_df.drop("脚質")
        today_df = today_df.join(final_df, on="馬名_ID", how="left")
        today_df = today_df.rename({"平均脚質": "脚質"})
        # 戦績データ取得
        today_df = PastData.get_aggregate_past_data(today_df, past_df, "")
        today_df = PastData.get_jockey_past_data(today_df, past_df, "")
        today_df = PastData.get_best_score(today_df, past_df, "")
        # 近5走のデータ取得
        past_df = result_df.filter(pl.col("馬名_ID").is_in(horse_ids) & (pl.col("日付") < date)).sort("日付", descending=True)
        past5_df = past_df.group_by("馬名_ID").head(5)
        today_df = PastData.get_aggregate_past_data(today_df, past5_df, "近5走")
        today_df = PastData.get_jockey_past_data(today_df, past5_df, "近5走")
        today_df = PastData.get_best_score(today_df, past5_df, "近5走")
        today_df = PastData.get_average_score(today_df, past5_df, 5)
        # 近3走のデータ取得
        past_df = result_df.filter(pl.col("馬名_ID").is_in(horse_ids) & (pl.col("日付") < date)).sort("日付", descending=True)
        past3_df = past_df.group_by("馬名_ID").head(3)
        today_df = PastData.get_aggregate_past_data(today_df, past3_df, "近3走")
        today_df = PastData.get_jockey_past_data(today_df, past3_df, "近3走")
        today_df = PastData.get_best_score(today_df, past3_df, "近3走")
        today_df = PastData.get_average_score(today_df, past3_df, 3)
        # 近5Rのデータマージ
        selected_columns = [
            '馬名_ID', '会場_ID', '開催数_ID', '開催日_ID', 'レース番号', 'レース名', '距離', '天気', 'レースタイプ', 'レース周り', '馬場', 'クラス', '頭数',
            '詳細条件_0', '詳細条件_1', '詳細条件_2', '詳細条件_3', '詳細条件_4',
            '4歳以上', '3歳以上', '3歳', '2歳', '同世代限定', '年齢条件', '馬齢', '指', '特指', '定量', '別定', 'ハンデ', '国際', '牡', '牝', '混', '見習騎手', '九州産馬',
            'ペース', 'ペース詳細', 'order', '枠番', '馬番', '斤量', '騎手_ID', '荒れ指数', 'タイム', 'ﾀｲﾑ指数', '1角', '4角', '位置取り', '脚質',
            '上り', '単勝', '人気', '馬体重', '備考', "出遅れ", "不利", '出走前馬レート', '出走前レース内馬レート', '出走後馬レート', '出走後レース内馬レート'
        ]
        # グループごとに行番号を付与
        df_with_index = past5_df.with_columns(
            pl.arange(0, pl.len()).over("馬名_ID").alias("row_num")
        )
        for i in range(5):
            group = df_with_index.filter(pl.col("row_num") == i).drop("row_num").select(selected_columns)
            group = group.rename({col: f"{i+1}走前_{col}" for col in selected_columns if col != "馬名_ID"})
            today_df = today_df.join(group, on="馬名_ID", how="left")
        # 騎手変更のフラグを作成
        today_df = today_df.with_columns(
            pl.when(pl.col("騎手_ID").is_null() | pl.col("1走前_騎手_ID").is_null())
            .then(False)
            .otherwise(pl.col("騎手_ID") != pl.col("1走前_騎手_ID"))
            .alias("騎手変更")
        )
        # 過去5走の騎手変更フラグ
        for i in range(1, 5):
            today_df = today_df.with_columns(
                pl.when(
                    pl.col(f"{i}走前_騎手_ID").is_null() | pl.col(f"{i+1}走前_騎手_ID").is_null()
                )
                .then(False)
                .otherwise(pl.col(f"{i}走前_騎手_ID") != pl.col(f"{i+1}走前_騎手_ID"))
                .alias(f"{i}走前_騎手変更")
            )
        # 5走前の騎手変更は常に False
        today_df = today_df.with_columns(pl.lit(False).alias("5走前_騎手変更"))
        # 斤量・距離関連のカラムを float に変換
        today_df = today_df.with_columns(
            pl.col("^.*斤量.*$").cast(pl.Float64),
            pl.col("^.*走前_距離.*$").cast(pl.Float64),
        )
        # 斤量差・距離差・クラス差の計算
        today_df = today_df.with_columns(
            pl.when(pl.col("斤量").is_null() | pl.col("1走前_斤量").is_null())
            .then(0)
            .otherwise(pl.col("斤量").cast(pl.Float64) - pl.col("1走前_斤量").cast(pl.Float64))
            .alias("前走との斤量差"),
            pl.when(pl.col("距離").is_null() | pl.col("1走前_距離").is_null())
            .then(0)
            .otherwise(pl.col("距離").cast(pl.Float64) - pl.col("1走前_距離").cast(pl.Float64))
            .alias("前走との距離差"),
            pl.when(pl.col("1走前_クラス").is_null() | pl.col("クラス").is_null())
            .then(pl.lit("不明"))
            .otherwise(pl.struct(["1走前_クラス", "クラス"]).map_elements(lambda x: PastData.cnv_class(x["1走前_クラス"], x["クラス"]), return_dtype=pl.Utf8))
            .alias("前走とのクラス差")
        )

        return today_df

    @staticmethod
    def create_past_data_extra(start: int, end: int, is_new=False) -> None:
        """ 過去データを生成する関数
            Args:
                start: 開始年
                end: 終了年
                is_new: 新規作成フラグ
            Returns:
                None
        """
        result_df = pl.read_parquet(PathManager.get_horse_rate_extra(False))
        base_df = pl.read_parquet(PathManager.get_horse_rate_extra(False))
        for year in tqdm(range(start, end)):
            # 土日情報取得
            target_df = base_df.filter(pl.col("日付").dt.year() == year)
            weeks = PastData.get_date_list(target_df["日付"].unique(maintain_order=True).to_list())
            past_df = pl.DataFrame()
            file_path = PathManager.get_past_table_extra(year, False)
            if os.path.isfile(file_path) and not is_new:
                past_df = pl.read_parquet(file_path)
            for days in tqdm(weeks):
                if len(past_df) > 0 and not is_new:
                    days_df = past_df.filter((pl.col("日付") >= days[0]) & (pl.col("日付") <= days[-1]))
                    if len(days_df) > 0:
                        print(f"continue {days[0]}, {days[-1]}")
                        continue
                print(f"{days[0]}, {days[-1]}")
                for day in days:
                    return_df = PastData.create_past_data_extra_core(days[0], day, target_df, result_df)
                    past_df = pl.concat([past_df, return_df])
            
            dir_path = Path(file_path).parent
            if not os.path.isdir(dir_path):
                print(f"Creating folder {dir_path}")
                os.makedirs(dir_path)
            past_df.write_parquet(file_path)

