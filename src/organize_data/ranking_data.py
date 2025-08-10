import polars as pl
from tqdm import tqdm
from src.utils.path_manager import PathManager


class RankingData:
    """ ランクデータを作成するクラス """
    @staticmethod
    def create_rank_data_extra_core(df: pl.DataFrame) -> pl.DataFrame:
        """ 各レース（race_id）内で特定の列のランクを計算する。
            Args:
                df (pl.DataFrame): DataFrame
            Returns:
                pl.DataFrame: ランクを追加したDataFrame
        """
        rank_cols = [col for col in df.columns if "勝率" in col or "連対率" in col or "複勝率" in col]
        df = df.with_columns(
            *[pl.col(col).rank(method='max', descending=True).over("race_id").alias(f'ランク_{col}') for col in rank_cols]
        )

        additional_ranks = [
            ('出走前馬レート', True), ('出走前レース内馬レート', True), ('調教タイム_1', False),
        ]
        df = df.with_columns(
            *[pl.col(col).rank(method='max', descending=desc).over("race_id").alias(f'ランク_{col}') for col, desc in additional_ranks]
        )

        for i in [5, 3]:
            ave = [
                (f'着順_{i}R', False), (f'ﾀｲﾑ指数_{i}R', False), (f'1角_{i}R', False), 
                (f'4角_{i}R', False), (f'位置取り_{i}R', False),(f'上り_{i}R', False), 
                (f'単勝_{i}R', False), (f'人気_{i}R', False), 
                (f'出走前馬レート_{i}R', True),(f'出走前レース内馬レート_{i}R', True), 
                (f'出走後馬レート_{i}R', True), (f'出走後レース内馬レート_{i}R', True)
            ]
            df = df.with_columns(
                *[pl.col(col).rank(method='max', descending=desc).over("race_id").alias(f'ランク_{col}') for col, desc in ave]
            )

        mid_cols = [col for col in df.columns if '中央値差分' in col]
        for col in mid_cols:
            if col in df.columns:
                if 'タイム_' in col or '上り' in col:
                    df = df.with_columns(
                        pl.col(col).rank(method='max', descending=False).over("race_id").alias(f'ランク_{col}')
                    )
                elif '出走前馬レート' in col or 'ﾀｲﾑ指数' in col:
                    df = df.with_columns(
                        pl.col(col).rank(method='max', descending=True).over("race_id").alias(f'ランク_{col}')
                    )

        return df
    
    @staticmethod
    def create_rank_data_extra(start: int, end: int) -> None:
        """ 統計テーブルからランクテーブルを作成する
            Args:
                start (int): 開始年
                end (int): 終了年
            Returns:
                None
        """
        for year in tqdm(range(start, end)):
            df = pl.read_parquet(PathManager.get_statistics_table_extra(year, False))
            df = RankingData.create_rank_data_extra_core(df)
            save_path=PathManager.get_rank_table_extra(year, False)
            df.write_parquet(save_path)

