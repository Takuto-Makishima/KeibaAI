import os
import pandas as pd
import polars as pl
from tqdm import tqdm
import trueskill

from src.utils.path_manager import PathManager
from src.table_data.refund_table import RefundTable


class HorseRateData:
    """ レートデータクラス """

    @staticmethod
    def create_data(end: int) -> None:
        """ ホースレート(trueskill)
        Args:
            end (int): 終了年
        Returns:
            None
        """
        result_df = []
        for year in tqdm(range(2008, end)):
            read_path = PathManager.get_merge_table_extra(year, False)
            add_df = pl.read_parquet(read_path)
            result_df.append(add_df)
        
        result_df = pl.concat(result_df)

        # `日付`、`race_id`、`order` の順に昇順ソート
        result_df = result_df.sort(['日付', 'race_id', 'order', "人気"], descending=[False, False, False, True])

        # 必要な列だけを抽出
        cols = ['race_id', 'order', '枠番', '馬番', '馬名_ID']
        df = result_df.select(cols).filter(pl.col('order') < 22)

        df = df.with_columns([
            pl.lit(None).alias("出走前馬レート"),
            pl.lit(None).alias("出走前レース内馬レート"),
            pl.lit(None).alias("出走後馬レート"),
            pl.lit(None).alias("出走後レース内馬レート"),
        ])
        # TrueSkill のパラメータ設定
        env = trueskill.TrueSkill(
            mu=25., sigma=25./3., beta=(25./3.)/2., tau=(25./3.)/100., draw_probability=0.001
        )

        race_list = df["race_id"].unique(maintain_order=True).to_list()
        uma_list = df["馬名_ID"].unique(maintain_order=True).to_list()

        # 各馬のレート初期化
        rate_dict = {k: (env.create_rating(), env.create_rating()) for k in uma_list}

        rate_before_, race_rate_before_, rate_after_, race_rate_after_ = [], [], [], []

        # 各レースごとの処理
        for race in tqdm(race_list):
            df_race = df.filter(pl.col("race_id") == race)
            race_vals = df_race.select(["race_id", "馬名_ID"]).rows()
            
            rate_before = [env.expose(rate_dict[el[1]][0]) for el in race_vals]
            rate_mean = sum(rate_before) / max(len(rate_before), 1)
            
            teams = [(rate_dict[el[1]][0],) for el in race_vals]
            teams = env.rate(teams, ranks=list(range(len(df_race))))
            rate_after = [env.expose(t[0]) for t in teams]
            
            race_rate_before = [
                r if rate_mean == 0 else (r - rate_mean) / rate_mean * 100
                for r in rate_before
            ]
            race_rate_after = [
                r if rate_mean == 0 else (r - rate_mean) / rate_mean * 100
                for r in rate_after
            ]
            for i, el in enumerate(race_vals):
                rate_dict[el[1]] = (teams[i][0], max(rate_dict[el[1]][0], teams[i][0]))

            rate_before_ += rate_before
            race_rate_before_ += race_rate_before
            race_rate_after_ += race_rate_after
            rate_after_ += rate_after
        mask = result_df["order"] < 22
        filtered_len = mask.sum()  # order < 22 の行数

        if len(rate_before_) == filtered_len:
            updates = pl.DataFrame({
                "race_id": result_df.filter(mask)["race_id"],  # `race_id` をキーにする
                "馬名_ID": result_df.filter(mask)["馬名_ID"],  # `馬名_ID` もキーにする
                "出走前馬レート": rate_before_,
                "出走前レース内馬レート": race_rate_before_,
                "出走後馬レート": rate_after_,
                "出走後レース内馬レート": race_rate_after_,
            })

            # `race_id` と `馬名_ID` をキーにしてマージ（`how="left"` で元のデータを保持）
            result_df = result_df.join(updates, on=["race_id", "馬名_ID"], how="left")

            # マージ後、欠損値（`null`）になった部分を 0.0 で埋める
            #result_df = result_df.fill_null(0.0)
        else:
            raise Exception()

        # order < 22 のデータを race_id ごとに統計量を計算
        grouped = (
            result_df.filter(pl.col("order") < 22)
            .group_by("race_id")
            .agg([
                pl.len().alias("出走前レース内馬レート_カウント"),
                pl.col("出走前レース内馬レート").mean().alias("出走前レース内馬レート_平均値"),
                pl.col("出走前レース内馬レート").median().alias("出走前レース内馬レート_中央値"),
                pl.col("出走前レース内馬レート").sum().alias("出走前レース内馬レート_合計"),
                pl.col("出走前レース内馬レート").std().alias("出走前レース内馬レート_標準偏差"),
                pl.col("出走前レース内馬レート").var().alias("出走前レース内馬レート_分散"),
                pl.col("出走前レース内馬レート").min().alias("出走前レース内馬レート_最小値"),
                pl.col("出走前レース内馬レート").max().alias("出走前レース内馬レート_最大値"),
            ])
        )

        # `race_id` をキーにして `result_df` に統計情報を結合（左結合）
        result_df = result_df.join(grouped, on="race_id", how="left")
        
        # 差分検出
        file_path = PathManager.get_horse_rate_extra(False)
        """temp_df = pl.DataFrame()
        if os.path.isfile(file_path):
            temp_df = pl.read_parquet(file_path)
        diff = pl.concat([merge_df, temp_df]).unique(keep='none')
        len_dif = len(diff)
        print(f'len_dif={len_dif}')"""
        merge_df = result_df.sort(['日付', '会場_ID', '開催数_ID', '開催日_ID', 'レース番号', '馬番'])
        merge_df.write_parquet(file_path)

