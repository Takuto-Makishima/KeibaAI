import os
import pandas as pd
import polars as pl
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import operator
from src.utils.path_manager import PathManager


class StaticticsData:
    """ 統計データを作成するクラス """
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
    def cal_finish_rate_by_condition(aggregate_df: pl.DataFrame, target_col: str, col_name: str, target_df: pl.DataFrame) -> pl.DataFrame:
        """ 指定された条件でグループ化し、着順の統計を計算する
            Args:
                aggregate_df: 集計データフレーム
                target_col: 対象のカラム名
                col_name: 統計を計算するカラム名
                target_df: 統計を追加するデータフレーム
            Returns:
                target_df: 統計を追加したデータフレーム
        """
        # グループ化のカラムパターン
        groupings = {
            "条件": ["レースタイプ", "会場_ID", "距離"],
            "完全同一条件": ["レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"],
            "馬場条件": ["レースタイプ", "会場_ID", "距離", "馬場"],
            "年齢上限条件": ["レースタイプ", "会場_ID", "距離", "年齢条件"],
            "クラス条件": ["レースタイプ", "会場_ID", "距離", "クラス"]
        }
        # 確率を計算する順位のしきい値（1着率・2着以内率・3着以内率）
        num_thresholds = [1, 2, 3]
        # 各グループごとに処理（col_name でグループ化）
        for key, group_cols in groupings.items():
            target_group_cols = group_cols + [target_col]
            for num in num_thresholds:
                name = ""
                if num == 1:
                    name = "勝利"
                elif num == 2:
                    name = "連対"
                elif num == 3:
                    name = "複勝"
                total_count_col = f"{col_name}_{key}別{name}総数"
                target_count_col = f"{col_name}_{key}別{name}数"
                if num == 1:
                    name = "勝"
                target_prob_col = f"{col_name}_{key}別{name}率"
                stats_df = (
                    aggregate_df
                    .group_by(target_group_cols)
                    .agg([
                        (pl.col("order") <= num).sum().alias(target_count_col),
                        pl.len().alias(total_count_col)  
                    ])
                    .with_columns(
                        (pl.col(target_count_col) / pl.col(total_count_col)).alias(target_prob_col)
                    )
                    .select(
                        target_group_cols + [total_count_col] + [target_count_col] + [target_prob_col]
                    )
                )
                target_df = target_df.join(stats_df, on=target_group_cols, how="left")

        return target_df

    @staticmethod
    def cal_finish_rate_by_category(aggregate_df: pl.DataFrame, target_col: str, col_name: str, target_df: pl.DataFrame) -> pl.DataFrame:
        """ 指定された列でグループ化し、着順の統計を計算する
            Args:
                aggregate_df: 集計データフレーム
                target_col: 対象のカラム名
                col_name: 統計を計算するカラム名
                target_df: 統計を追加するデータフレーム
            Returns:
                target_df: 統計を追加したデータフレーム
        """
        num_thresholds = [1, 2, 3]
        cols = ["レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"]
        for col in cols:
            group_cols = [col, target_col]
            for num in num_thresholds:
                name = ""
                if num == 1:
                    name = "勝利"
                elif num == 2:
                    name = "連対"
                elif num == 3:
                    name = "複勝"
                total_count_col = f"{col_name}_{col}別{name}総数"
                target_count_col = f"{col_name}_{col}別{name}数"
                if num == 1:
                    name = "勝"
                target_prob_col = f"{col_name}_{col}別{name}率"
                stats_df = (
                    aggregate_df
                    .group_by(group_cols)
                    .agg([
                        # 各 num_thresholds の○着以内の数
                        (pl.col("order") <= num).sum().alias(target_count_col),
                        # グループ内の総数
                        pl.len().alias(total_count_col)  
                    ])
                    .with_columns(
                        (pl.col(target_count_col) / pl.col(total_count_col))
                        .alias(target_prob_col)
                    )
                    .select(
                        group_cols + [total_count_col] + [target_count_col] + [target_prob_col]
                    )
                )
                # target_df にマージ（LEFT JOIN）
                target_df = target_df.join(stats_df, on=group_cols, how="left")

        return target_df

    @staticmethod
    def cal_mode(aggregate_df: pl.DataFrame, target_col: str, col_name: str, target_df: pl.DataFrame) -> pl.DataFrame:
        """ 指定された列でグループ化し、最頻値を計算する
            Args:
                aggregate_df: 集計データフレーム
                target_col: 対象のカラム名
                col_name: 統計を計算するカラム名
                target_df: 統計を追加するデータフレーム
            Returns:
                target_df: 統計を追加したデータフレーム
        """
        # グループ化するカラムのリスト
        groupings = {
            "条件": ["レースタイプ", "会場_ID", "距離"],
            "完全同一条件": ["レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"],
            "馬場条件": ["レースタイプ", "会場_ID", "距離", "馬場"],
            "年齢上限条件": ["レースタイプ", "会場_ID", "距離", "年齢条件"],
            "クラス条件": ["レースタイプ", "会場_ID", "距離", "クラス"]
        }

        for key, group_cols in groupings.items():
            for num in [1, 2, 3]:  # 1着以内, 2着以内, 3着以内
                name = ""
                if num == 1:
                    name = "勝利"
                elif num == 2:
                    name = "連対"
                elif num == 3:
                    name = "複勝"
                filtered_df = aggregate_df.filter(pl.col("order") <= num)
                
                mode_df = (
                    filtered_df
                    .group_by(group_cols)
                    .agg(pl.col(target_col).mode().first().alias(f"{col_name}_{key}_{name}馬_最頻値"))
                )

                # target_df にマージ（LEFT JOIN）
                target_df = target_df.join(mode_df, on=group_cols, how="left")

        return target_df

    @staticmethod
    def cal_difference_from_the_standard_value(df: pl.DataFrame) -> None:
        """ 指定された列の中央値との差分を計算する
            Args:
                df: データフレーム
            Returns:
                df: 中央値との差分を追加したデータフレーム
        """
        spans = ["", "近5走", "近3走"]
        groupings = [ "完全同一条件", "同一条件", "会場距離条件", "馬場条件", "年齢条件", "クラス条件", "距離条件" ]
        num_thresholds = [0, 1, 2, 3]
        cols = ["タイム", "上り", "出走前馬レート", "ﾀｲﾑ指数"]
        for span in spans:
            for key in groupings:
                for num in num_thresholds:
                    for col in cols:
                        if num == 0:
                            name = ""
                        elif num == 1:
                            name = "勝利"
                        elif num == 2:
                            name = "連対"
                        elif num == 3:
                            name = "複勝"
                
                        df = df.with_columns(
                            pl.when(
                                pl.col(f"{span}{key}ベスト{col}").is_not_null() & pl.col(f"{col}_{key}別{name}中央値").is_not_null()
                            )
                            .then(pl.col(f"{span}{key}ベスト{col}") - pl.col(f"{col}_{key}別{name}中央値"))
                            .otherwise(None)
                            .alias(f"{span}{key}{col}_{name}中央値差分")
                        )
        return df

    @staticmethod
    def cal_statistics(aggregate_df: pl.DataFrame, target_col: str, target_df: pl.DataFrame, num_thresholds = [0, 1, 2, 3]) -> pl.DataFrame:
        """ 指定された列でグループ化し、統計量を計算する
            memo: タイム等のデータなので、レースタイプ、会場、距離、は同じものでグループ化する必要がある。
            Args:
                aggregate_df: 集計データフレーム
                target_col: 対象のカラム名
                target_df: 統計を追加するデータフレーム
                num_thresholds: 着順のしきい値リスト
            Returns:
                target_df: 統計を追加したデータフレーム
        """
        groupings = {
            "完全同一条件": ["レースタイプ", "会場_ID", "距離", "馬場", "年齢条件", "クラス"],
            "同一条件": ["レースタイプ", "会場_ID", "距離", "馬場"],
            "会場距離条件": ["レースタイプ", "会場_ID", "距離"],
            "馬場条件": ["レースタイプ", "距離", "馬場"],
            "年齢条件": ["レースタイプ", "距離", "年齢条件"],
            "クラス条件": ["レースタイプ", "距離", "クラス"],
            "距離条件": ["レースタイプ", "距離"],
        }
        
        for key, group_cols in groupings.items():
            for num in num_thresholds:
                if num == 0:
                    name = ""
                    filtered_df = aggregate_df
                elif num == 1:
                    name = "勝利"
                    filtered_df = aggregate_df.filter(pl.col("order") <= 1)
                elif num == 2:
                    name = "連対"
                    filtered_df = aggregate_df.filter(pl.col("order") <= 2)
                elif num == 3:
                    name = "複勝"
                    filtered_df = aggregate_df.filter(pl.col("order") <= 3)
                
                stats_df = (
                    filtered_df
                    .group_by(group_cols)
                    .agg([
                        pl.len().alias(f"{target_col}_{key}別{name}総数"),
                        pl.col(target_col).mean().alias(f"{target_col}_{key}別{name}平均値"),
                        pl.col(target_col).median().alias(f"{target_col}_{key}別{name}中央値"),
                        pl.col(target_col).std().alias(f"{target_col}_{key}別{name}標準偏差"),
                        pl.col(target_col).var().alias(f"{target_col}_{key}別{name}分散"),
                        pl.col(target_col).min().alias(f"{target_col}_{key}別{name}最小値"),
                        pl.col(target_col).max().alias(f"{target_col}_{key}別{name}最大値"),
                    ])
                )
                target_df = target_df.join(stats_df, on=group_cols, how="left")
        
        return target_df

    @staticmethod
    def create_statistics_data_core_extra(temp_df: pl.DataFrame, date: datetime, target_df: pl.DataFrame) -> pl.DataFrame:
        """ 統計データを作成するコア処理
            Args:
                temp_df: 集計データフレーム
                date: 日付
                target_df: 統計を追加するデータフレーム
            Returns:
                target_df: 統計を追加したデータフレーム
        """
        year, month = date.year, date.month
        start_date = datetime(year - 5, month, 1)
        # order < 20 のフィルタリング
        temp_df = temp_df.filter(pl.col("order") < 20)
        # 指定期間のデータを取得
        aggregate_df = temp_df.filter((pl.col("日付") >= start_date) & (pl.col("日付") < date))
        # ソート処理
        aggregate_df = aggregate_df.sort(["日付", "race_id", "馬番"])
        # `会場_ID` のユニークな値を取得
        places = target_df["会場_ID"].unique(maintain_order=True).to_list()
        # `会場_ID` が `places` に含まれるデータのみ取得
        aggregate_df = aggregate_df.filter(pl.col("会場_ID").is_in(places))

        # レースの統計 f"{target_col}_{key}別{name}中央値"
        target_df = StaticticsData.cal_statistics(aggregate_df, "タイム", target_df)
        target_df = StaticticsData.cal_statistics(aggregate_df, '上り', target_df)
        target_df = StaticticsData.cal_statistics(aggregate_df, '出走前馬レート', target_df)
        target_df = StaticticsData.cal_statistics(aggregate_df, 'ﾀｲﾑ指数', target_df)
        target_df = StaticticsData.cal_statistics(aggregate_df, '荒れ指数', target_df, [0])
        # 差分を計算 f"{span}{key}{col}_{name}中央値差分"
        target_df = StaticticsData.cal_difference_from_the_standard_value(target_df)
        # 最頻値 f"{col_name}_{key}_{name}馬_最頻値"
        target_df = StaticticsData.cal_mode(aggregate_df, "ペース", "ペース", target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '前走との斤量差', '前走との斤量差', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '前走との距離差', '前走との距離差', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '前走とのクラス差', '前走とのクラス差', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '枠番', '枠番', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, 'peds_00_ID', "父", target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, 'peds_32_ID', "母父", target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, 'peds_type', 'peds_type', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '脚質', '脚質', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '脚色', '脚色', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '評価', '評価', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '評価欄', '評価欄', target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '騎手_ID', "騎手", target_df)
        target_df = StaticticsData.cal_mode(aggregate_df, '調教師_ID', '調教師', target_df)
        # 指定した列のレート計算 
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, '騎手_ID', "騎手", target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, '調教師_ID', '調教師', target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, 'peds_00_ID', "父", target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, 'peds_32_ID', "母父", target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, 'peds_type', 'peds_type', target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, '脚色', '脚色', target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, '評価', '評価', target_df)
        target_df = StaticticsData.cal_finish_rate_by_category(aggregate_df, '評価欄', '評価欄', target_df)
        # 指定した条件のレート計算
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '人気', '人気', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '枠番', '枠番', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '騎手_ID', '騎手', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, 'peds_00_ID', "父", target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, 'peds_32_ID', "母父", target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, 'peds_type', 'peds_type', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '脚質', '脚質', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '脚色', '脚色', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '評価', '評価', target_df)
        target_df = StaticticsData.cal_finish_rate_by_condition(aggregate_df, '評価欄', '評価欄', target_df)
        
        return target_df

    @staticmethod
    def init(year: int) -> pl.DataFrame:
        start = year - 6
        end = year + 1
        years = []
        for i in range(start, end):
            path = PathManager.get_past_table_extra(i, False)
            if os.path.isfile(path) == True:
                years.append(pl.read_parquet(path))
        temp_df = pl.concat(years)
        return temp_df

    @staticmethod
    def create_statictics_data_extra(start_year: int, end_year: int, is_new: bool=False) -> None:
        """ 統計データを作成する
            Args:
                start_year: 開始年
                end_year: 終了年
                is_new: 新規作成フラグ
            Returns:
                None
        """
        for year in tqdm(range(start_year, end_year)):
            temp_df = StaticticsData.init(year)

            result_df = pl.DataFrame()
            file_path = PathManager.get_statistics_table_extra(year, False)
            if os.path.isfile(file_path) and not is_new:
                result_df = pl.read_parquet(file_path)

            year_df = temp_df.filter(temp_df["日付"].dt.year() == year)
            weeks = StaticticsData.get_date_list(year_df["日付"].unique(maintain_order=True))
            for days in tqdm(weeks):
                target_days_df = pl.DataFrame()
                if len(result_df) != 0 and not is_new:
                    target_days_df = result_df.filter((pl.col("日付") >= days[0]) & (pl.col("日付") <= days[-1]))
                    if len(target_days_df) > 0 and not is_new:
                        print(f'continue {days[0]}, {days[-1]}')
                        continue
                print(f"{days[0]}, {days[-1]}")
                target_days_df = year_df.filter(year_df["日付"].is_between(days[0], days[-1]))
                
                return_df = StaticticsData.create_statistics_data_core_extra(temp_df, days[0], target_days_df)
                result_df = pl.concat([result_df, return_df])

            save_path = PathManager.get_statistics_table_extra(year, False)
            result_df.write_parquet(save_path)


