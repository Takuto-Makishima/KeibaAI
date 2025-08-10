from datetime import datetime


class PathManager:
    """ パス管理クラス """
    FEATURES_COUNT = 500
    #METRIC = 'PR_AUC'
    METRIC = 'f1_score'
    #METRIC = 'precision'
    #METRIC = 'focal_loss'

    @staticmethod
    def get_html_race_table(year: int, is_pickle: bool = True) -> str:
        """ レーステーブルのパスを取得する
            Args:
                year(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                str: レーステーブルのパス
        """
        if is_pickle:
            return f'./html/race_table/{year}.pickle'
        else:
            return f'./html/race_table/{year}.parquet'
    
    @staticmethod
    def get_html_pedigree_table():
        """ 血統テーブルのパスを取得する
            Returns:
                str: 血統テーブルのパス
        """
        return './html/pedigree_table/all/pedigree_table.pickle'

    @staticmethod
    def get_html_training_table(year: int):
        """ 調教テーブルのパスを取得する
            Args:
                year(int): 年
            Returns:
                str: 調教テーブルのパス
        """
        return f'./html/training_table/{year}.pickle'

    @staticmethod
    def get_race_table_extra(year: int, is_pickle: bool = True) -> str:
        """ レーステーブルのパスを取得する
            Args:
                year(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                str: レーステーブルのパス
        """
        if is_pickle:
            return f'./html/data/preprocessing/preprocessing_race_table_extra_{year}.pickle'
        else:
            return f'./html/data/preprocessing/preprocessing_race_table_extra_{year}.parquet'

    @staticmethod
    def get_pedigree_table(is_pickle: bool = True) -> str:
        """ 血統テーブルのパスを取得する
            Args:
                is_pickle(bool): pickle形式か
            Returns:
                str: 血統テーブルのパス
        """
        if is_pickle:
            return './html/data/preprocessing/preprocessing_pedigree_table.pickle'
        else:
            return './html/data/preprocessing/preprocessing_pedigree_table.parquet'

    @staticmethod
    def get_training_table_extra(year: int, is_pickle: bool = True) -> str:
        """ 調教テーブルのパスを取得する
            Args:
                year(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                str: 調教テーブルのパス
        """
        if is_pickle:
            return f'./html/data/preprocessing/preprocessing_training_table_extra_{year}.pickle'
        else:
            return f'./html/data/preprocessing/preprocessing_training_table_extra_{year}.parquet'

    @staticmethod
    def get_merge_table_extra(year: int, is_pickle: bool = True) -> str:
        """ マージテーブルのパスを取得する
            Args:
                year(int): 年
            Returns:
                str: マージテーブルのパス
        """
        if is_pickle:
            return f'./html/data/merge/merge_table_extra_{year}.pickle'
        else:
            return f'./html/data/merge/merge_table_extra_{year}.parquet'

    @staticmethod
    def get_horse_rate_extra(is_pickle: bool = True) -> str:
        """ 馬レートのパスを取得する
            Args:
                is_pickle(bool): pickle形式か
            Returns:
                str: 馬レートのパス
        """
        if is_pickle:
            return './html/data/horse_rate/horse_rate_true_skill_all_extra.pickle'
        else:
            return './html/data/horse_rate/horse_rate_true_skill_all_extra.parquet'

    @staticmethod
    def get_past_table_extra(year: int, is_pickle: bool = True) -> str:
        """ 過去レーステーブルのパスを取得する
            Args:
                year(int): 年
                is_pickle
            Returns:
                str: 過去レーステーブルのパス
        """
        if is_pickle:
            return f'./html/data/past_extra/past_table_extra_{year}.pickle'
        else:
            return f'./html/data/past_extra/past_table_extra_{year}.parquet'

    @staticmethod
    def get_statistics_table_extra(year: int, is_pickle: bool = True) -> str:
        """ 統計テーブルのパスを取得する
            Args:
                year(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                str: 統計テーブルのパス
        """
        if is_pickle:
            return f'./html/data/statistics_extra/statistics_table_extra_{year}.pickle'
        else:
            return f'./html/data/statistics_extra/statistics_table_extra_{year}.parquet'

    @staticmethod
    def get_rank_table_extra(year: int, is_pickle: bool = True) -> str:
        """ ランクテーブルのパスを取得する
            Args:
                year(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                str: ランクテーブルのパス
        """
        if is_pickle:
            return f'./html/data/rank/rank_table_extra_{year}.pickle'
        else:
            return f'./html/data/rank/rank_table_extra_{year}.parquet'

    @staticmethod
    def get_dataset_extra(year: int, is_pickle: bool = True) -> str:
        """ 学習データのパスを取得する
            Args:
                year(int): 年
                is_pickle(bool): pickle形式か
            Returns:
                str: 学習データのパス
        """
        if is_pickle:
            return f'./html/data/dataset_extra/dataset_extra_{year}.pickle'
        else:
            return f'./html/data/dataset_extra/dataset_extra_{year}.parquet'

    @staticmethod
    def get_selected_features(start: datetime, end: datetime, target: str, race_type: str, place: str, distance: int) -> str:
        """ 特徴量のパスを取得する
            Args:
                start(datetime): 開始日
                end(datetime): 終了日
                target(str): ターゲット
                race_type(str): レース種別
                place(str): 場所
                distance(int): 距離
            Returns:
                str: 特徴量のパス
        """
        str_date = f'{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}'
        if (distance==0) and (place == ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/selected_features.csv'
        elif (distance!=0) and (place == ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{distance}/selected_features.csv'
        elif (distance==0) and (place != ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{place}/selected_features.csv'
        else:
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{place}/{distance}/selected_features.csv'

    @staticmethod
    def get_selected_features_from_str(str_date: str, target: str, race_type: str, place: str, distance: int) -> str:
        """ 特徴量のパスを取得する
            Args:
                str_date(str): 日付
                target(str): ターゲット
                race_type(str): レース種別
                place(str): 場所
                distance(int): 距離
            Returns:
                str: 特徴量のパス
        """
        if (distance==0) and (place == ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/selected_features.csv'
        elif (distance!=0) and (place == ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{distance}/selected_features.csv'
        elif (distance==0) and (place != ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{place}/selected_features.csv'
        else:
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{place}/{distance}/selected_features.csv'

    @staticmethod
    def get_average_features(str_date: str, target: str, race_type: str, place: str, distance: int) -> str:
        """ 平均特徴量のパスを取得する
            Args:
                str_date(str): 日付
                target(str): ターゲット
                race_type(str): レース種別
                place(str): 場所
                distance(int): 距離
            Returns:
                str: 平均特徴量のパス
        """
        if (distance==0) and (place == ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/average_features.pickle'
        elif (distance!=0) and (place == ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{distance}/average_features.pickle'
        elif (distance==0) and (place != ''):
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{place}/average_features.pickle'
        else:
            return f'./html/models/{str_date}/selected_features/{target}/{race_type}/{place}/{distance}/average_features.pickle'
    
    @staticmethod
    def get_param_path(start: datetime, end: datetime, objective: str, target: str, race_type: str, place: str, distance: int) -> str:
        """ パラメータのパスを取得する
            Args:
                start(datetime): 開始日
                end(datetime): 終了日
                objective(str): 目的
                target(str): ターゲット
                race_type(str): レース種別
                place(str): 場所
                distance(int): 距離
            Returns:
                str: パラメータのパス
        """
        str_date = f'{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}'
        if (distance==0) and (place == ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/hyperparameter.txt'
        elif (distance!=0) and (place == ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{distance}/hyperparameter.txt'
        elif (distance==0) and (place != ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{place}/hyperparameter.txt'
        else:
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{place}/{distance}/hyperparameter.txt'

    @staticmethod
    def get_eval_path(start: datetime, end: datetime, objective: str, target: str, race_type: str, place: str, distance: int) -> str:
        """ 評価のパスを取得する
            Args:
                start(datetime): 開始日
                end(datetime): 終了日
                objective(str): 目的
                target(str): ターゲット
                race_type(str): レース種別
                place(str): 場所
                distance(int): 距離
            Returns:
                str: 評価のパス
        """
        str_date = f'{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}'
        if (distance==0) and (place == ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/evaluation.pickle'
        elif (distance!=0) and (place == ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{distance}/evaluation.pickle'
        elif (distance==0) and (place != ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{place}/evaluation.pickle'
        else:
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{place}/{distance}/evaluation.pickle'

    @staticmethod
    def get_model_path(start: datetime, end: datetime, objective: str, target: str, race_type: str, place: str, distance: int) -> str:
        """ モデルのパスを取得する
            Args:
                start(datetime): 開始日
                end(datetime): 終了日
                objective(str): 目的
                target(str): ターゲット
                race_type(str): レース種別
                place(str): 場所
                distance(int): 距離
            Returns:
                str: モデルのパス
        """
        str_date = f'{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}'
        if (distance==0) and (place == ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/model.pickle'
        elif (distance!=0) and (place == ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{distance}/model.pickle'
        elif (distance==0) and (place != ''):
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{place}/model.pickle'
        else:
            return f'./html/models/{str_date}/{objective}/{target}/{race_type}/{place}/{distance}/model.pickle'

    @staticmethod
    def get_predict_dataset_path(today: datetime, race_id: str) -> str:
        """ 予測データセットのパスを取得する
            Args:
                today(datetime): 日付
                race_id(str): レースID
            Returns:
                str: 予測データセットのパス
        """
        str_day = datetime.datetime.strftime(today, '%Y%m%d')
        return f'./html/predict/{str_day}/{race_id}/predict_dataset.pickle'
