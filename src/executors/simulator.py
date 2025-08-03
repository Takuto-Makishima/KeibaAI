import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

class Simulator:

    def __init__():
        pass

    @staticmethod
    def add_place_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        target = 'refund_place'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                if np.isnan(horse_1) == True:
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    if (horse_1 == int(win)):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_win_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        target = 'refund_win'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                if np.isnan(horse_1) == True:
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    if (horse_1 == int(win)):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_wide_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        target = 'refund_wide'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                if (np.isnan(float(horse_1)) == True) or (np.isnan(float(horse_2)) == True):
                    continue
                # win列取得
                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)           
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']
                    if (horse_1 in new_list) and (horse_2 in new_list):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_bracket_quinella_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        target = 'refund_bracket_quinella'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_f_1'])
                horse_2 = float(pred.loc[idx, 'p_f_2'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True):
                    continue
                # win列取得
                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)           
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']
                    if ((horse_1 == new_list[0]) and (horse_2 == new_list[1])) or ((horse_1 == new_list[1]) and (horse_2 == new_list[0])):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_quinella_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        target = 'refund_quinella'
        pred[target] = np.nan
        for idx in pred.index:
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']
                    if ((horse_1 == new_list[0]) and (horse_2 == new_list[1])) or ((horse_1 == new_list[1]) and (horse_2 == new_list[0])):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100
        return pred

    @staticmethod
    def add_exacta_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        taeget = 'refund_exacta'
        pred[taeget] = np.nan
        for idx in pred.index:
            # 単勝
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']                   
                    if (new_list[0] == horse_1) and (new_list[1] == horse_2):
                        pred.loc[idx,taeget] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,taeget] = -100

        return pred

    @staticmethod
    def add_trio_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        target = 'refund_trio'
        pred[target] = np.nan
        for idx in pred.index:
            # 単勝
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                horse_3 = float(pred.loc[idx, 'p_h_3'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True) or (np.isnan(horse_3) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                horse_3 = int(horse_3)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']                   
                    if (horse_1 in new_list) and (horse_2 in new_list) and (horse_3 in new_list):
                        pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,target] = -100

        return pred

    @staticmethod
    def add_tierce_result(tbl_pred, refund_table):
        pred = tbl_pred.copy()
        taeget = 'refund_tierce'
        pred[taeget] = np.nan
        for idx in pred.index:
            # 単勝
            if idx in refund_table.index:
                horse_1 = float(pred.loc[idx, 'p_h_1'])
                horse_2 = float(pred.loc[idx, 'p_h_2'])
                horse_3 = float(pred.loc[idx, 'p_h_3'])
                if (np.isnan(horse_1) == True) or (np.isnan(horse_2) == True) or (np.isnan(horse_3) == True):
                    continue

                cnt = len(refund_table.filter(like='win').columns)
                horse_1 = int(horse_1)
                horse_2 = int(horse_2)
                horse_3 = int(horse_3)
                for i in range(0, cnt):
                    win = refund_table.loc[idx, f'win_{i}']
                    if type(win) == type(None):
                        continue
                    # 1列内に空白で区切られた文字(数値)がある為、分割する
                    win = win.split(' ')
                    # 空白を削除しつつ変換を行う
                    new_list = [int(item) for item in win if item != '']                   
                    if (new_list[0] == horse_1) and (new_list[1] == horse_2) and (new_list[2] == horse_3):
                        pred.loc[idx,taeget] = (int(refund_table.loc[idx,f'ref_{i}']) - 100)
                        break
                    else:
                        pred.loc[idx,taeget] = -100

        return pred

    @staticmethod
    def calculation_data(tbl):
        lst = ['refund_place','refund_win','refund_wide','refund_bracket_quinella','refund_quinella','refund_exacta','refund_trio','refund_tierce']
        result = None
        for col in lst:
            tbl_ref = tbl.copy()
            tbl_ref = tbl_ref[tbl_ref[col].notna()]
            tbl_ref[col] = tbl_ref[col].astype(int)
            # 総レース数
            total_count = len(tbl)
            #　購入数
            buy = len(tbl_ref[tbl_ref[col].notna()])
            # 購入率
            buy_rate = buy / total_count
            # 的中
            hits = tbl_ref[tbl_ref[col] != -100][col]
            # 的中数(払戻数)
            hit_count = len(hits)
            # 的中率
            hit_rate = hit_count / buy
            # 購入金額
            purchase_price = buy * 100
            # 払戻金額
            refund = (hits + 100).sum()
            # 収支
            total = tbl_ref[col].sum()
            # 回収率
            recovery_rate = refund / purchase_price

            lst = [f'{total_count}R', f'{buy}R', f'{buy_rate*100:.1f}%', f'{hit_count}', f'{hit_rate*100:.1f}%',f'{purchase_price}円', f'{refund}円', f'{total}円', f'{recovery_rate*100:.1f}%']

            df = pd.DataFrame(lst, index =['総レース数', '購入数', '購入率', '的中数', '的中率', '購入金額', '払戻金額', '収支', '回収率'],columns =[col.replace('refund_', '')])
            if type(result) != None:
                result = pd.concat([result, df], axis=1)
            else:
                result = df
        return result

    @staticmethod
    def show_graph(pred_tbl, lst=['refund_place','refund_win','refund_wide','refund_bracket_quinella','refund_quinella','refund_exacta','refund_trio','refund_tierce']):
        plt.figure(figsize=(15,20))
        import matplotlib.colors as mcolors
        row = 4
        col  = 2
        cnt = 1
        unique_dates = pred_tbl['日付'].dropna().sort_values().unique()
        colors = list(mcolors.TABLEAU_COLORS.values())
        pred = pred_tbl.copy()
        print('プロット')
        for columns in tqdm(lst):
            pred = pred[pred[columns].notna()]
            plt.subplot(row, col, cnt, title=columns.replace('refund_',''))
            plt.grid(axis='y')
            pred[f'{columns}_sum'] = pred[columns].cumsum()
            plt.plot(pred[f'{columns}_sum'])
            cnt = cnt + 1
            # 背景に日付ごとの範囲を塗る
            for i, date in enumerate(unique_dates):
                date_data = pred[pred['日付'] == date]
            
                start_idx = date_data.index[0]
                end_idx = date_data.index[-1]
                plt.axvspan(start_idx, end_idx, color=colors[i % len(colors)], alpha=0.3)
        #グラフを表示
        plt.show()

    @staticmethod
    def add_combo_box(tbl_pred, target, table, pattern, ext):
        pred = tbl_pred.copy()
        target = target
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        for idx, row in tbl_pred[tbl_pred['p_h_1'].notna()][pattern].iterrows():
            # idx 存在確認
            if idx not in table.index:
                continue
            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(table.filter(like='win').columns)
            for i in range(0, cnt):
                win = table.loc[idx, f'win_{i}']
                if (type(win) == type(None)) or (win == 0):
                    continue

                if type(win) == str:
                    win = win.split(' ')
                    win = [x for x in win if x != '']
                    win = set([int(x) for x in win])

                for item in itertools.combinations(row, ext):
                    if isinstance(win,np.int32) or isinstance(win,np.int64):
                        item = int(item[0])
                    else:
                        item = set([int(x) for x in item])

                    if item == win:
                        #print(idx, item, win)
                        pred.loc[idx,target] += (int(table.loc[idx,f'ref_{i}']))
                        pred.loc[idx,f'{target}_hit'] = True
                        break

        return pred

    @staticmethod
    def lists_match(l1, l2):
        if len(l1) != len(l2):
            return False
        return all(x == y and type(x) == type(y) for x, y in zip(l1, l2))

    @staticmethod
    def add_perm_box(tbl_pred, target, table, pattern, ext):
        pred = tbl_pred.copy()
        target = target
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        for idx, row in tbl_pred[tbl_pred['p_h_1'].notna()][pattern].iterrows():
            # idx 存在確認
            if idx not in table.index:
                continue

            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(table.filter(like='win').columns)
            for i in range(0, cnt):
                win = table.loc[idx, f'win_{i}']
                if (type(win) == type(None)) or (win == 0):
                    continue

                win = win.split(' ')
                win = [x for x in win if x != '']
                win = [int(x) for x in win]

                for item in itertools.permutations(row, ext):
                    item = [int(x) for x in item]
                    if Simulator.lists_match(win, item):
                        #print(idx, item, win)
                        pred.loc[idx,target] += (int(table.loc[idx,f'ref_{i}']))
                        pred.loc[idx,f'{target}_hit']=True
                        break

        return pred

    @staticmethod
    def add_exacta_formation(tbl_pred, refund_table, lst, pattern):
        pred = tbl_pred.copy()
        target = 'refund_exacta_formation'
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        # 予測テーブルループ
        for idx in pred.index:
            # idx 存在確認
            if idx not in refund_table.index:
                continue
            if pd.isna(tbl_pred.loc[idx, 'p_h_1']):
                continue

            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(refund_table.filter(like='win').columns)
            for num in range(0, cnt):
                win = refund_table.loc[idx, f'win_{num}']
                if (type(win) == type(None)) or (pd.isnull(pred.loc[idx, lst[0]].iloc[0])):
                    continue
                win = win.split(' ')
                win = [x for x in win if x != '']
                if (int(win[0]) in pred.loc[idx, lst[0]].to_list()) and (int(win[1]) in pred.loc[idx, lst[1]].to_list()):
                    pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{num}']))
                    pred.loc[idx,f'{target}_hit']=True

        return pred

    @staticmethod
    def add_tierce_formation(tbl_pred, refund_table, lst, pattern):
        pred = tbl_pred.copy()
        target = 'refund_tierce_formation'
        pred[target] = np.nan
        pred[f'{target}_hit'] = False

        # 予測テーブルループ
        for idx in pred.index:
            # idx 存在確認
            if idx not in refund_table.index:
                continue
            if pd.isna(tbl_pred.loc[idx, 'p_h_1']):
                continue

            # 購入金額は自動集計する為、払戻のみ記録できるようにする
            pred.loc[idx, target] = 0

            # win列のみ抽出
            cnt = len(refund_table.filter(like='win').columns)
            for num in range(0, cnt):
                win = refund_table.loc[idx, f'win_{num}']
                if (type(win) == type(None)) or (pd.isnull(pred.loc[idx, lst[0]].iloc[0])):
                    continue
                win = win.split(' ')
                win = [x for x in win if x != '']
                if (int(win[0]) in pred.loc[idx, lst[0]].to_list()) and (int(win[1]) in pred.loc[idx, lst[1]].to_list()) and (int(win[2]) in pred.loc[idx, lst[2]].to_list()):
                    pred.loc[idx,target] = (int(refund_table.loc[idx,f'ref_{num}']))
                    pred.loc[idx,f'{target}_hit']=True

        return pred

    @staticmethod
    def calcuration(table, col, count):
        tbl = table.copy()
        # 総レース数
        total_count = len(tbl)

        tbl = tbl[tbl[col].notna()]
        tbl[col] = tbl[col].astype(int)

        #　購入数
        buy = len(tbl)
        # 購入率
        buy_rate = buy / total_count

        # 的中
        hits = tbl[tbl[f'{col}_hit'] == True][col]
        # 的中数(払戻数)
        hit_count = len(hits)
        # 的中率
        hit_rate = hit_count / buy

        # 購入金額
        purchase_price = buy * (100 * count)
        # 払戻金額
        refund = int(hits.sum())
        # 収支
        total = refund - purchase_price
        # 回収率
        recovery_rate = refund / purchase_price

        lst = [f'{total_count}R', f'{buy}R', f'{buy_rate*100:.1f}%', f'{hit_count}', f'{hit_rate*100:.1f}%',f'{purchase_price}円', f'{refund}円', f'{total}円', f'{recovery_rate*100:.1f}%']

        df = pd.DataFrame(lst, index =['総レース数', '購入数', '購入率', '的中数', '的中率', '購入金額', '払戻金額', '収支', '回収率'],columns =[col.replace('refund_', '')])
            
        return df

    @staticmethod
    def create_predict_table(result_df, count, exists_past_3rd, exists_past_2nd, threshold,
                              is_today=False, is_expected= False, target_col='pred', is_sort=True):
        columns = []
        columns.append('日付')
        columns.append('レースタイプ')        
        columns.append('会場')
        columns.append('距離')
        columns.append('馬場')

        for i in range(0, count):
            columns.append(f'p_f_{i+1}')
            columns.append(f'p_h_{i+1}')
            columns.append(f'p_p_{i+1}')
            columns.append(f'pred_{i+1}')

        race_ids = result_df["race_id"].unique()
        tbl_pred = pd.DataFrame(data=[], index=race_ids, columns=columns)

        day = '日付'
        place = '会場'
        race_type = 'レースタイプ'
        distance = '距離'
        grade = 'クラス'
        frame_num = '枠番'
        horse_num = '馬番'
        popularity = '人気'
        ground = '馬場'

        for race_id in tqdm(race_ids):
            target_df = result_df.query("race_id == @race_id").copy()#loc[result_df["race_id"]==race_id].copy()
            target_day = target_df[day].iloc[0]
            target_place = target_df[place].iloc[0]
            target_type = target_df[race_type].iloc[0]
            target_distance = target_df[distance].iloc[0]
            target_class = target_df[grade].iloc[0]
            target_ground = target_df[ground].iloc[0]
            
            if target_df[target_col].isna().any():
                print('continue', race_id, target_place, target_type[0], int(target_distance), target_class)
                continue

            if exists_past_3rd == True:
                past_3rd = target_df['3走前_着順'].notna().all()
                if past_3rd == False:
                    print('continue', race_id, target_place, target_type[0], int(target_distance), target_class)
                    continue

            if exists_past_2nd == True:
                past_2nd = target_df['2走前_着順'].notna().all()
                if past_2nd == False:
                    print('continue', race_id, target_place, target_type[0], int(target_distance), target_class)
                    continue

            disp_df = None
            if is_expected == True:
                target_df['標準化'] = target_df[target_col].transform(lambda x: ((x - x.mean()) / x.std()))
                target_df['偏差値'] = target_df[target_col].transform(lambda x: ((x - x.mean()) / x.std()) * 10 + 50)
                target_df['期待値'] = target_df['偏差値'] * target_df['単勝']

            if is_sort==True:
                disp_df = target_df.sort_values(target_col, ascending=False)
            else:
                disp_df = target_df.copy()

            # 指定列に値が入っていることを確認
            # count_above_zero = disp_df[target_col].apply(lambda x: x > threshold).sum()
            # is_pred_col = False
            # if count_above_zero < count:
            #     print(f'continue under count {count_above_zero} < {count}', race_id, target_place, target_type[0], int(target_distance), target_class)
            #     continue

            # 上位取得
            temp = disp_df.head(5).copy()
            value = temp[target_col].iloc[-1]
            cnt = 1
            for index, frame, horse, prd, dis, ev_id in zip(temp["race_id"], temp[frame_num], temp[horse_num], temp[target_col], temp[distance], temp[place]):
                tbl_pred.loc[index,'日付'] = target_day
                tbl_pred.loc[index,'レースタイプ'] = target_type
                tbl_pred.loc[index,'会場'] = target_place
                tbl_pred.loc[index,'距離'] = target_distance
                tbl_pred.loc[index,'馬場'] = target_ground
                tbl_pred.loc[index,f'p_f_{cnt}'] = frame
                tbl_pred.loc[index,f'p_h_{cnt}'] = horse
                tbl_pred.loc[index,f'p_p_{cnt}'] = 0
                tbl_pred.loc[index,f'pred_{cnt}'] = prd
                cnt += 1

        return tbl_pred
    
    @staticmethod
    def show_graph_with_date_highlight(pred_tbl, 
                                       lst=['refund_place', 'refund_win', 'refund_wide', 
                                            'refund_bracket_quinella', 'refund_quinella', 
                                            'refund_exacta', 'refund_trio', 'refund_tierce'], 
                                       date_column='日付'):
        import matplotlib.colors as mcolors
        
        plt.figure(figsize=(15, 20))
        row, col, cnt = 4, 2, 1
        pred = pred_tbl.copy()
        
        # 日付のユニーク値を取得（順序を保つ）
        unique_dates = pred[date_column].dropna().sort_values().unique()
        
        # 背景色に使用する色リスト
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        print('プロット')
        for column in tqdm(lst):
            pred = pred[pred[column].notna()].copy()
            
            # 累積和を計算
            pred[f'{column}_sum'] = pred[column].cumsum()
            
            plt.subplot(row, col, cnt, title=column.replace('refund_', ''))
            plt.grid(True)
            
            # 背景に日付ごとの範囲を塗る
            for i, date in enumerate(unique_dates):
                date_data = pred[pred[date_column] == date]
                if not date_data.empty:
                    start_idx = date_data.index.min()
                    end_idx = date_data.index.max()
                    plt.axvspan(start_idx, end_idx, color=colors[i % len(colors)], alpha=0.1)
            
            
            plt.legend(fontsize=8, loc='upper left')
            cnt += 1
    
        # グラフを表示
        plt.tight_layout()
        plt.show()
