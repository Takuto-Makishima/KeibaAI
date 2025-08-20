import os
from datetime import datetime
import pandas as pd
import polars as pl
from tqdm import tqdm
import itertools

from src.creators.dataset import Dataset
from src.creators.ai_creator import AiCreator
from src.executors.simulator import Simulator


if __name__ == '__main__':

    START_YEAR = 2025
    END_YEAR = 2026
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #objective = 'lambdarank'  # 'rank_xendcg', 'lambdarank'

    CREATE_LIST = [
        # [datetime(2025,1,5).date(), datetime(2025,1,6).date()],
        # [datetime(2025,1,11).date(), datetime(2025,1,12).date(), datetime(2025,1,13).date()],
        # [datetime(2025,1,18).date(), datetime(2025,1,19).date()],
        # [datetime(2025,1,25).date(), datetime(2025,1,26).date()],
        # [datetime(2025,2,1).date(), datetime(2025,2,2).date()],
        # [datetime(2025,2,8).date(), datetime(2025,2,9).date(), datetime(2025,2,10).date()],
        # [datetime(2025,2,15).date(), datetime(2025,2,16).date()],
        # [datetime(2025,2,22).date(), datetime(2025,2,23).date()],
        # [datetime(2025,3,1).date(), datetime(2025,3,2).date()],
        # [datetime(2025,3,8).date(), datetime(2025,3,9).date()],
        # [datetime(2025,3,15).date(), datetime(2025,3,16).date()],
        # [datetime(2025,3,22).date(), datetime(2025,3,23).date()],
        # [datetime(2025,3,29).date(), datetime(2025,3,30).date()],
        # [datetime(2025,4,5).date(), datetime(2025,4,6).date()],
        # [datetime(2025,4,12).date(), datetime(2025,4,13).date()],
        # [datetime(2025,4,19).date(), datetime(2025,4,20).date()],
        # [datetime(2025,4,26).date(), datetime(2025,4,27).date()],
        # [datetime(2025,5,3).date(), datetime(2025,5,4).date()],
        # [datetime(2025,5,10).date(), datetime(2025,5,11).date()],
        # [datetime(2025,5,17).date(), datetime(2025,5,18).date()],
        # [datetime(2025,5,24).date(), datetime(2025,5,25).date()],
        # [datetime(2025,5,31).date(), datetime(2025,6,1).date()],
        # [datetime(2025,6,7).date(), datetime(2025,6,8).date()],
        # [datetime(2025,6,14).date(), datetime(2025,6,15).date()],
        [datetime(2025,6,21).date(), datetime(2025,6,22).date()],
        [datetime(2025,6,28).date(), datetime(2025,6,29).date()],
        [datetime(2025,7,5).date(), datetime(2025,7,6).date()],
        [datetime(2025,7,12).date(), datetime(2025,7,13).date()],
        [datetime(2025,7,19).date(), datetime(2025,7,20).date()],
        # [datetime(2025,7,26).date(), datetime(2025,7,27).date()],
        # [datetime(2025,8,2).date(), datetime(2025,8,3).date()],
    ]
    # Dataset.create_dataset(CREATE_LIST, START_YEAR, END_YEAR)

    for objective in ['rank_xendcg', 'lambdarank']:
        # # モデル作成
        # AiCreator.create_rank_model(CREATE_LIST, 2010, END_YEAR, objective)

        # 予測データ作成
        Simulator.create_prediction_data(CREATE_LIST, START_YEAR, END_YEAR, True, objective)

        # # レース結果取得
        # Simulator.get_race_result(CREATE_LIST, objective)

        # 予測結果集計
        Simulator.predict_result_aggregate(CREATE_LIST, objective)

    # # 表示データ作成
    # target_value = 'rank3'
    # count = 5
    # exists_past_3rd = False
    # exists_past_2nd = False
    # threshold = 0.0
    # is_today = True
    # is_expected = False
    # is_sort = True
    # for objective in ['rank_xendcg', 'lambdarank']:
    #     pred_result = Simulator.read_predict_result(CREATE_LIST, objective, target_value)
    #     refund_table_df = Simulator.read_refund_data(CREATE_LIST)

    #     tbl_pred = Simulator.create_summary_table(pred_result, count, exists_past_3rd, exists_past_2nd, threshold, is_today, is_expected, 'pred', is_sort)
    #     tbl_pred = Simulator.add_place_result(tbl_pred, refund_table_df.place)
    #     tbl_pred = Simulator.add_win_result(tbl_pred, refund_table_df.win)
    #     tbl_pred = Simulator.add_wide_result(tbl_pred, refund_table_df.wide)
    #     tbl_pred = Simulator.add_bracket_quinella_result(tbl_pred, refund_table_df.bracket_quinella)
    #     tbl_pred = Simulator.add_quinella_result(tbl_pred, refund_table_df.quinella)
    #     tbl_pred = Simulator.add_exacta_result(tbl_pred, refund_table_df.exacta)
    #     tbl_pred = Simulator.add_trio_result(tbl_pred, refund_table_df.trio)
    #     tbl_pred = Simulator.add_tierce_result(tbl_pred, refund_table_df.tierce)
    #     df = Simulator.calculation_data(tbl_pred)
    #     print(df)
    #     Simulator.show_graph(tbl_pred)

    #     disp_df = pd.DataFrame()
    #     tbl_pred, disp_df = Simulator.process_combo_box_bet(tbl_pred, disp_df, refund_table_df.place, 'refund_place_box', ['p_h_1', 'p_h_2'], 1)
    #     tbl_pred, disp_df = Simulator.process_combo_box_bet(tbl_pred, disp_df, refund_table_df.win, 'refund_win_box', ['p_h_1', 'p_h_2'], 1)
    #     tbl_pred, disp_df = Simulator.process_combo_box_bet(tbl_pred, disp_df, refund_table_df.wide, 'refund_wide_box', ['p_h_1', 'p_h_2', 'p_h_3'], 2)
    #     tbl_pred, disp_df = Simulator.process_combo_box_bet(tbl_pred, disp_df, refund_table_df.bracket_quinella, 'refund_bracket_quinella_box', ['p_h_1', 'p_h_2', 'p_h_3'], 2)
    #     tbl_pred, disp_df = Simulator.process_combo_box_bet(tbl_pred, disp_df, refund_table_df.quinella, 'refund_quinella_box', ['p_h_1', 'p_h_2', 'p_h_3'], 2)
    #     tbl_pred, disp_df = Simulator.process_prem_box_bet(tbl_pred, disp_df, refund_table_df.exacta, 'refund_exacta_box', ['p_h_1', 'p_h_2', 'p_h_3'], 2)
    #     tbl_pred, disp_df = Simulator.process_combo_box_bet(tbl_pred, disp_df, refund_table_df.trio, 'refund_trio_box', ['p_h_1', 'p_h_2', 'p_h_3', 'p_h_4'], 3)
    #     tbl_pred, disp_df = Simulator.process_prem_box_bet(tbl_pred, disp_df, refund_table_df.tierce, 'refund_tierce_box', ['p_h_1', 'p_h_2', 'p_h_3', 'p_h_4'], 3)
    #     print(disp_df)
    #     Simulator.show_graph(tbl_pred, ['refund_place_box','refund_win_box','refund_wide_box','refund_bracket_quinella_box','refund_quinella_box','refund_exacta_box','refund_trio_box','refund_tierce_box'])

    #     disp_df = pd.DataFrame()
    #     tbl_pred, disp_df = Simulator.process_formation_bet(tbl_pred, disp_df, refund_table_df.exacta, 'refund_exacta_formation', ['p_h_1'], ['p_h_1', 'p_h_2', 'p_h_3'])
    #     tbl_pred, disp_df = Simulator.process_formation_bet(tbl_pred, disp_df, refund_table_df.tierce, 'refund_tierce_formation', ['p_h_1', 'p_h_2'], ['p_h_1', 'p_h_2', 'p_h_3'], ['p_h_1', 'p_h_2', 'p_h_3', 'p_h_4'])
    #     print(disp_df)
    #     Simulator.show_graph(tbl_pred, ['refund_exacta_formation', 'refund_tierce_formation'])
