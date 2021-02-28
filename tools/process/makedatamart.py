#ライブラリをインポートする。
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

import copy
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import gc

def makelgbmtraindf(racecard_fn, raceinfo_fn, training_fn):

	#出馬表ファイルをデータフレーム化する。
	racecard = pd.read_csv(racecard_fn, encoding='cp932')
	
	#年情報に2000を加算する。
	#マージする際の変数として利用するために事前に加工する。
	racecard['年'] = racecard['年'] + 2000

	#レース単位で斤量差を考慮した変数を作成する。
	#効果がありそうなため作成する。
	weight_mst = racecard.groupby(['年','月','日','場所','レース番号'])[['斤量']].max()
	weight_mst = weight_mst.reset_index()
	weight_mst = weight_mst.rename(columns={'斤量' : '最大斤量'})
	racecard = pd.merge(racecard, weight_mst, on=['年','月','日','場所','レース番号'], how='inner')
	racecard['斤量差'] = racecard['最大斤量'] - racecard['斤量']

	#レース結果ファイルをデータフレーム化する。
	raceinfo = pd.read_csv(raceinfo_fn, encoding='cp932')

	raceinfo = raceinfo[raceinfo['場所'] != 'None']
	raceinfo = raceinfo.reset_index()
	raceinfo.drop('index', axis=1, inplace=True)

	#新馬戦学習モデル用のデータフレームを作成する。(makerefunddf関数は下で作成している。)
	refund_df, reward_df = makerefunddf(raceinfo)

	#同着レース判定用データフレームを作成する。
	same_race_df = refund_df[['年', '月', '日', '場所', 'レース番号', 'same_race_flg']]

	#新馬時調教ファイルをデータフレーム化する。
	debut_training = pd.read_csv(training_fn, encoding='cp932')

	#集約変数を設定する。
	#集約の定義関数を作成する。
	aggs = {}
	gb_var = ['Time1', 'Time2', 'Time3', 'Time4', 'Lap1', 'Lap2', 'Lap3', 'Lap4']

	for var in gb_var:
		aggs[var] = ['max', 'min', 'mean', 'count']

	def create_new_columns(aggs):
		return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

	#新馬時調教ファイルを馬名で集約する。
	new_columns = create_new_columns(aggs)
	debut_training_gb = debut_training.groupby('馬名').agg(aggs)
	debut_training_gb.columns = new_columns
	debut_training_gb.reset_index(drop=False, inplace=True)

	#出馬用ファイル、レース結果ファイル、新馬時調教ファイルを結合したデータフレームを作成する。
	train_df = pd.merge(racecard, raceinfo, on=['年','月','日','場所','レース番号'], how='inner')
	train_df = pd.merge(train_df, debut_training_gb, on='馬名', how='left')
	train_df = pd.merge(train_df, reward_df, on=['年','月','日','場所','レース番号','馬番'], how='left')
	train_df = pd.merge(train_df, same_race_df, on=['年','月','日','場所','レース番号'], how='inner')
	win_odds_std_df = train_df.groupby(['年','月','日','場所','レース番号'])[['単勝オッズ']].std()
	win_odds_std_df = win_odds_std_df.reset_index()
	win_odds_std_df = win_odds_std_df.rename(columns={'単勝オッズ':'単勝オッズ_std'})
	train_df = pd.merge(train_df, win_odds_std_df, on=['年','月','日','場所','レース番号'], how='inner')
	train_df['target'] = train_df['target'] / np.log(train_df['単勝オッズ']) * np.log(train_df['人気'] + 0.1)
	train_df['target'] = train_df['target'].fillna(0)

	#同着があったレースを除外する。
	train_df = train_df[train_df['same_race_flg'] == 0]

	return train_df

def makemultiregtraindf(racecard_fn, raceinfo_fn, training_fn):

	#出馬表ファイルをデータフレーム化する。
	racecard = pd.read_csv(racecard_fn, encoding='cp932')
	
	#年情報に2000を加算する。
	#マージする際の変数として利用するために事前に加工する。
	racecard['年'] = racecard['年'] + 2000

	#レース結果ファイルをデータフレーム化する。
	raceinfo = pd.read_csv(raceinfo_fn, encoding='cp932')

	#新馬戦学習モデル用のデータフレームを作成する。(makerefunddf関数は下で作成している。)
	refund_df, reward_df = makerefunddf(raceinfo)

	#同着レース判定用データフレームを作成する。
	same_race_df = refund_df[['年', '月', '日', '場所', 'レース番号', 'same_race_flg']]

	#出馬用ファイル、レース結果ファイルを結合したデータフレームを作成する。
	train_df = pd.merge(racecard, raceinfo, on=['年','月','日','場所','レース番号'], how='inner')
	train_df = pd.merge(train_df, reward_df, on=['年','月','日','場所','レース番号','馬番'], how='left')
	train_df = pd.merge(train_df, refund_df, on=['年','月','日','場所','レース番号'], how='left')
	train_df = pd.merge(train_df, same_race_df, on=['年','月','日','場所','レース番号'], how='inner')
	train_df['target'] = train_df['target'] / np.log(train_df['単勝オッズ']) * np.log(train_df['人気'] + 0.1)
	train_df['target'] = train_df['target'].fillna(0)
	
	#同着があったレースを除外する。
	#train_df = train_df[train_df['same_race_flg'] == 0]
	debut_train_df = train_df[train_df['クラス名'] == '新馬']
	debut_train_df = debut_train_df[['場所', '父馬名', '母馬名', '母の父馬名', '父タイプ', '母父タイプ', '芝ダ', '距離', '馬場状態']]

	train_df['カウントフラグ'] = 1
	train_df['単勝フラグ'] = 0
	train_df['単勝払戻金'] = 0
	train_df.loc[(train_df['馬番'] == train_df['win_number']), '単勝フラグ'] = 100
	train_df.loc[(train_df['馬番'] == train_df['win_number']), '単勝払戻金'] = train_df['win_reward']

	train_df['複勝フラグ'] = 0
	train_df['複勝払戻金'] = 0
	train_df.loc[(train_df['馬番'] == train_df['place_1st_number']), '複勝フラグ'] = 100
	train_df.loc[(train_df['馬番'] == train_df['place_2nd_number']), '複勝フラグ'] = 100
	train_df.loc[(train_df['馬番'] == train_df['place_3rd_number']), '複勝フラグ'] = 100
	train_df.loc[(train_df['馬番'] == train_df['place_1st_number']), '複勝払戻金'] = train_df['place_1st_reward']
	train_df.loc[(train_df['馬番'] == train_df['place_2nd_number']), '複勝払戻金'] = train_df['place_2nd_reward']
	train_df.loc[(train_df['馬番'] == train_df['place_3rd_number']), '複勝払戻金'] = train_df['place_3rd_reward']

	train_df = train_df[['年','月','日','レース番号', 'クラス名', '馬名','人気','単勝オッズ','場所',
	'父馬名','母の父馬名','父タイプ','母父タイプ','騎手','調教師','馬主','生産者',
	'芝ダ','距離', '馬場状態', 'カウントフラグ','単勝フラグ', '単勝払戻金', '複勝フラグ', '複勝払戻金', 'target']]

	#各集約変数毎に勝率、複勝率を算出し、サンプル数を加味する。
	def make_base_arrange_df(df):

	    def sigmoid(x, k):
	        return 1 / (1 + np.exp(- x / k))

	    gb_cols = ['父馬名', '母の父馬名', '父タイプ', '母父タイプ', '騎手', '調教師', '馬主', '生産者']
	    target_cols = ['単勝フラグ', '複勝フラグ','単勝払戻金', '複勝払戻金']

	    for col in gb_cols:
	        for target in target_cols:
	        #スムージングのハイパーパラメータを設定する。
	            k = 100
	            n_i = df.groupby(col).count()[target]
	            lambda_n_i = sigmoid(n_i, k)
	            n_i_mean = df.groupby(col).mean()[target]
	            N_target = df[target].sum() / 100
	            N = len(df)
	            all_mean = N_target / N
	            temp_dict = pd.DataFrame(lambda_n_i * n_i_mean + (1 - lambda_n_i) * all_mean)
	            temp_dict = temp_dict.reset_index()
	            temp_dict.index = temp_dict[col].values
	            temp_dict = temp_dict[target].to_dict()
	            df[col+target+'_全体平均'] = df[col].map(temp_dict)
	    return df

	train_df = make_base_arrange_df(train_df)

	#各集約変数毎にレース時属性を追加して、勝率、複勝率を集約する。
	def make_gb_df(var1, df):
	    gb_mean_df = df.groupby([var1,'場所','芝ダ','距離','馬場状態']).mean()[['単勝フラグ', '複勝フラグ', '単勝払戻金', '複勝払戻金']]
	    gb_mean_df = gb_mean_df.reset_index()
	    gb_mean_df = gb_mean_df.rename(columns={'単勝フラグ': var1 + '単勝フラグ_集約平均', '複勝フラグ': var1 + '複勝フラグ_集約平均', '単勝払戻金': var1 + '単勝払戻金_集約平均', '複勝払戻金': var1 + '複勝払戻金_集約平均'})
	    gb_sum_df = df.groupby([var1,'場所','芝ダ','距離','馬場状態']).sum()['カウントフラグ']
	    gb_sum_df = gb_sum_df.reset_index()
	    gb_sum_df = gb_sum_df.rename(columns={'カウントフラグ' : var1 + '集約回数'})
	    gb_df = pd.merge(gb_mean_df, gb_sum_df, on=[var1,'場所','芝ダ','距離','馬場状態'], how='inner')
	    return gb_df

	gb_df1 = make_gb_df('父馬名', train_df)
	gb_df2 = make_gb_df('母の父馬名', train_df)
	gb_df3 = make_gb_df('父タイプ', train_df)
	gb_df4 = make_gb_df('母父タイプ', train_df)
	gb_df5 = make_gb_df('騎手', train_df)
	gb_df6 = make_gb_df('調教師', train_df)
	gb_df7 = make_gb_df('馬主', train_df)
	gb_df8 = make_gb_df('生産者', train_df)

	train_df = pd.merge(train_df, gb_df1, on=['父馬名','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df2, on=['母の父馬名','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df3, on=['父タイプ','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df4, on=['母父タイプ','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df5, on=['騎手','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df6, on=['調教師','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df7, on=['馬主','場所','芝ダ','距離','馬場状態'], how='inner')
	train_df = pd.merge(train_df, gb_df8, on=['生産者','場所','芝ダ','距離','馬場状態'], how='inner')

	#各集約変数毎にレース時属性を追加して、勝率、複勝率を集約したものにサンプル数を加味する。
	def make_base_arrange_df(df):
	    gb_cols = ['父馬名', '母の父馬名', '父タイプ', '母父タイプ', '騎手', '調教師', '馬主', '生産者']
	    target_cols = ['単勝フラグ', '複勝フラグ','単勝払戻金', '複勝払戻金']
	    for col in gb_cols:
	        for target in target_cols:
	            #スムージングのハイパーパラメータを設定する。
	            k = 100
	            df[col+'加重度'] = 1 / (1 + np.exp( -df[col+'集約回数'] / k))
	            df[col+target+'_調整平均'] = df[col+'加重度']*df[col+target+'_集約平均'] + (1 - df[col+'加重度'])*df[col+target+'_全体平均']
	            del df[col+target+'_集約平均']
	            del df[col+target+'_全体平均']
	        del df[col+'加重度']
	        del df[col+'集約回数']
	    return df

	train_df = make_base_arrange_df(train_df)

	drop_col = ['カウントフラグ','単勝フラグ', '単勝払戻金', '複勝フラグ', '複勝払戻金']
	train_df.drop(drop_col, axis=1, inplace=True)

	return train_df

def makeanalyticsdf(racecard_fn, raceinfo_fn):

	#出馬表ファイルをデータフレーム化する。
	racecard = pd.read_csv(racecard_fn, encoding='cp932')
	
	#年情報に2000を加算する。
	#マージする際の変数として利用するために事前に加工する。
	racecard['年'] = racecard['年'] + 2000

	#レース単位で斤量差を考慮した変数を作成する。
	#効果がありそうなため作成する。
	weight_mst = racecard.groupby(['年','月','日','場所','レース番号'])[['斤量']].max()
	weight_mst = weight_mst.reset_index()
	weight_mst = weight_mst.rename(columns={'斤量' : '最大斤量'})
	racecard = pd.merge(racecard, weight_mst, on=['年','月','日','場所','レース番号'], how='inner')
	racecard['斤量差'] = racecard['最大斤量'] - racecard['斤量']

	#レース結果ファイルをデータフレーム化する。
	raceinfo = pd.read_csv(raceinfo_fn, encoding='cp932')

	#新馬戦学習モデル用のデータフレームを作成する。(makerefunddf関数は下で作成している。)
	refund_df, reward_df = makerefunddf(raceinfo)

	#同着レース判定用データフレームを作成する。
	#same_race_df = refund_df[['年', '月', '日', '場所', 'レース番号', 'same_race_flg']]

	#出馬用ファイル、レース結果ファイルを結合したデータフレームを作成する。
	analytics_df = pd.merge(racecard, raceinfo, on=['年','月','日','場所','レース番号'], how='inner')
	analytics_df = pd.merge(analytics_df, refund_df, on=['年','月','日','場所','レース番号'], how='inner')
	
	#同着があったレースを除外する。
	analytics_df = analytics_df[analytics_df['same_race_flg'] == 0]
	
	analytics_df = analytics_df[['年','月','日','場所','レース番号','馬名','性別','年齢','父馬名','母馬名','母の父馬名','父タイプ','母父タイプ',
	'馬体重','騎手','調教師','馬主','生産者','斤量','馬番','確定着順','人気','単勝オッズ','1角通過順','2角通過順','3角通過順','4角通過順',
	'脚質','クラス名','芝ダ','距離','頭数','コーナー回数','天候','馬場状態',
	'win_number','win_reward','place_1st_number','place_1st_reward','place_2nd_number','place_2nd_reward','place_3rd_number','place_3rd_reward',
	'quinella_1','quinella_2','quinella_reward','exacta_1st','exacta_2nd','exacta_reward',
	'quinella_place_1st_1','quinella_place_1st_2','quinella_place_1st_reward',
	'quinella_place_2nd_1','quinella_place_2nd_2','quinella_place_2nd_reward',
	'quinella_place_3rd_1','quinella_place_3rd_2','quinella_place_3rd_reward',
	'trio_1','trio_2','trio_3','trio_reward','trifecta_1','trifecta_2','trifecta_3','trifecta_reward',
	'1st_reward','2nd_reward','3rd_reward','上がり3F_x' ,'Ave-3F' ,'コース区分',
	'Lap1','Lap2','Lap3','Lap4','Lap5','Lap6','Lap7','Lap8','Lap9','Lap10','Lap11','Lap12','Lap13',
	'Lap14','Lap15','Lap16','Lap17','Lap18','Lap19','Lap20','Lap21','Lap22','Lap23','Lap24','Lap25',
	'前後3F差','1着入線タイム','1-5着平均タイム','2-5着平均タイム','25%平均タイム','通過3F']]

	return analytics_df

def makelgbmpredictdf(racecard_fn, training_fn):
	
	#出馬表ファイルをデータフレーム化する。
	racecard = pd.read_csv(racecard_fn, encoding='cp932')

	racecard[' 単勝'] = racecard[' 単勝'].astype('str')
	racecard = racecard[racecard[' 単勝'] != '取消し']

	racecard.drop('所属', axis=1, inplace=True)
	racecard = racecard.rename(columns={'R':'レース番号', '芝・ダート':'芝ダ', '馬場状態(暫定)':'馬場状態', '天候(暫定)':'天候', 
										'  馬名':'馬名', '父':'父馬名', '母':'母馬名', '母父':'母の父馬名', 
										' 馬主':'馬主' , ' 生産者':'生産者', '所属.1':'所属', ' 単勝':'単勝オッズ', '番':'馬番'})

	racecard['単勝オッズ'] = racecard['単勝オッズ'].astype('float')
	
	#レース単位で斤量差を考慮した変数を作成する。
	#効果がありそうなため作成する。
	weight_mst = racecard.groupby(['年','月','日','場所','レース番号'])[['斤量']].max()
	weight_mst = weight_mst.reset_index()
	weight_mst = weight_mst.rename(columns={'斤量' : '最大斤量'})
	racecard = pd.merge(racecard, weight_mst, on=['年','月','日','場所','レース番号'], how='inner')
	racecard['斤量差'] = racecard['最大斤量'] - racecard['斤量']

	#非根幹距離フラグを作成する。
	#効果がありそうなため作成する。
	racecard['非根幹フラグ'] = racecard['距離'].apply(lambda x : 1 if (x % 400) != 0 else 0)

	win_odds_std_df = racecard.groupby(['年','月','日','場所','レース番号'])[['単勝オッズ']].std()
	win_odds_std_df = win_odds_std_df.reset_index()
	win_odds_std_df = win_odds_std_df.rename(columns={'単勝オッズ':'単勝オッズ_std'})
	racecard = pd.merge(racecard, win_odds_std_df, on=['年','月','日','場所','レース番号'], how='inner')

	#新馬時調教ファイルをデータフレーム化する。
	debut_training = pd.read_csv(training_fn, encoding='cp932')

	#集約変数を設定する。
	#集約の定義関数を作成する。
	aggs = {}
	gb_var = ['Time1', 'Time2', 'Time3', 'Time4', 'Lap1', 'Lap2', 'Lap3', 'Lap4']

	for var in gb_var:
		aggs[var] = ['max', 'min', 'mean', 'count']

	def create_new_columns(aggs):
		return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

	#新馬時調教データを馬名で集約する。
	new_columns = create_new_columns(aggs)
	debut_training_gb = debut_training.groupby('馬名').agg(aggs)
	debut_training_gb.columns = new_columns
	debut_training_gb.reset_index(drop=False, inplace=True)

	#出馬表データと新馬時調教データを結合する。
	predict_df = pd.merge(racecard, debut_training_gb, on='馬名', how='left')

	#オブジェクト型変数をカテゴリ値に変換する。
	datamart_object_col_list = predict_df.select_dtypes(include=['object'])
	for var in datamart_object_col_list:
		predict_df[var] = predict_df[var].astype('category')

	return predict_df

def makemultiregpredictdf(racecard_fn, training_fn, train_df):
	
	#出馬表ファイルをデータフレーム化する。
	racecard = pd.read_csv(racecard_fn, encoding='cp932')

	racecard[' 単勝'] = racecard[' 単勝'].astype('str')
	racecard = racecard[racecard[' 単勝'] != '取消し']

	racecard.drop('所属', axis=1, inplace=True)
	racecard = racecard.rename(columns={'R':'レース番号', '芝・ダート':'芝ダ', '馬場状態(暫定)':'馬場状態', '天候(暫定)':'天候', 
										'  馬名':'馬名', '父':'父馬名', '母':'母馬名', '母父':'母の父馬名', 
										' 馬主':'馬主' , ' 生産者':'生産者', '所属.1':'所属', ' 単勝':'単勝オッズ', '番':'馬番'})

	racecard['単勝オッズ'] = racecard['単勝オッズ'].astype('float')

	def make_countresult_mst(df, var):
		col_list = ['場所','芝ダ','距離','馬場状態']
		col_list.append(var)
		col_list.append(var+'単勝払戻金_調整平均')
		col_list.append(var+'複勝払戻金_調整平均')
		col_list.append(var+'単勝フラグ_調整平均')
		col_list.append(var+'複勝フラグ_調整平均')
		mst = df[col_list]
		mst = mst[~mst.duplicated()]
		return mst

	mst1 = make_countresult_mst(train_df, var='父馬名')
	mst2 = make_countresult_mst(train_df, var='母の父馬名')
	mst3 = make_countresult_mst(train_df, var='父タイプ')
	mst4 = make_countresult_mst(train_df, var='母父タイプ')
	mst5 = make_countresult_mst(train_df, var='騎手')
	mst6 = make_countresult_mst(train_df, var='調教師')
	mst7 = make_countresult_mst(train_df, var='馬主')
	mst8 = make_countresult_mst(train_df, var='生産者')

	predict_df = pd.merge(racecard, mst1, on=['父馬名','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst2, on=['母の父馬名','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst3, on=['父タイプ','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst4, on=['母父タイプ','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst5, on=['騎手','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst6, on=['調教師','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst7, on=['馬主','場所','芝ダ','距離','馬場状態'], how='left')
	predict_df = pd.merge(predict_df, mst8, on=['生産者','場所','芝ダ','距離','馬場状態'], how='left')

	predict_df = predict_df.fillna(10)
	
	return predict_df

def makerefunddf(df):

	#この関数で目的変数を作成している。

	#コース区分の情報の欠損値を埋める。
	df['コース区分'] = df['コース区分'].fillna('Z')

	#非根幹距離フラグを作成する。
	#効果がありそうなため作成する。
	df['非根幹フラグ'] = df['距離'].apply(lambda x : 1 if (x % 400) != 0 else 0)

	#単勝配当から情報を抽出するために加工する。
	win_str_list = []
	for i in range(len(df)):
		win_str_list.append(repr(df['単勝配当'][i]))

	#単勝配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	win_str_list_df = pd.DataFrame(win_str_list)
	win_str_list_df = win_str_list_df.rename(columns={0:'win_result'})
	win_str_list_df['win_result'] = win_str_list_df['win_result'].str.replace("'","")
	win_str_list_df['win_result'] = win_str_list_df['win_result'].str.split('\\')
	win_result_list = list(win_str_list_df['win_result'])

	win_result_temp_df = pd.DataFrame(win_result_list)
	win_result_temp_df.drop([1,3,4], axis=1, inplace=True)

	win_number_df = win_result_temp_df[[0]]
	win_number_df = win_number_df.rename(columns={0:'win_number'})

	win_reward_df = pd.DataFrame(list(win_result_temp_df[2].str.split('/')))
	win_reward_df['same_win_flg'] = win_reward_df[1].apply(lambda x : 0 if x == None else 1)
	win_reward_df = win_reward_df.rename(columns={0:'win_reward'})
	win_reward_df.drop([1], axis=1, inplace=True)

	win_result_df = pd.merge(win_number_df, win_reward_df, left_index=True, right_index=True)

	win_result_df['win_number'] = win_result_df['win_number'].astype('int')
	win_result_df['win_reward'] = win_result_df['win_reward'].astype('float')

	#複勝配当から払い戻しの情報を抽出するために加工する。
	place_str_list = []
	for i in range(len(df)):
		place_str_list.append(repr(df['複勝配当'][i]))

	#複勝配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	place_str_list_df = pd.DataFrame(place_str_list)
	place_str_list_df = place_str_list_df.rename(columns={0:'place_result'})
	place_str_list_df['place_result'] = place_str_list_df['place_result'].str.split('/')
	place_result_list = list(place_str_list_df['place_result'])
	place_result_df = pd.DataFrame(place_result_list)
	place_result_df = place_result_df.fillna('0\\ \\100')
	place_result_df[0] = place_result_df[0].str.replace("'","")
	place_result_df[1] = place_result_df[1].str.replace("'","")
	place_result_df[2] = place_result_df[2].str.replace("'","")
	place_result_df[3] = place_result_df[3].str.replace("'","")

	place_1st_result_list = list(place_result_df[0].str.split('\\'))
	place_1st_result_df = pd.DataFrame(place_1st_result_list)
	place_1st_result_df = place_1st_result_df.rename(columns={0:'place_1st_number'})
	place_1st_result_df = place_1st_result_df.rename(columns={2:'place_1st_reward'})
	place_1st_result_df.drop([1], axis=1, inplace=True)

	place_2nd_result_list = list(place_result_df[1].str.split('\\'))
	place_2nd_result_df = pd.DataFrame(place_2nd_result_list)
	place_2nd_result_df = place_2nd_result_df.rename(columns={0:'place_2nd_number'})
	place_2nd_result_df = place_2nd_result_df.rename(columns={2:'place_2nd_reward'})
	place_2nd_result_df.drop([1], axis=1, inplace=True)

	place_3rd_result_list = list(place_result_df[2].str.split('\\'))
	place_3rd_result_df = pd.DataFrame(place_3rd_result_list)
	place_3rd_result_df = place_3rd_result_df.rename(columns={0:'place_3rd_number'})
	place_3rd_result_df = place_3rd_result_df.rename(columns={2:'place_3rd_reward'})
	place_3rd_result_df.drop([1], axis=1, inplace=True)

	place_4th_result_list = list(place_result_df[3].str.split('\\'))
	place_4th_result_df = pd.DataFrame(place_4th_result_list)
	place_4th_result_df = place_4th_result_df.rename(columns={0:'place_4th_number'})
	place_4th_result_df = place_4th_result_df.rename(columns={2:'place_4th_reward'})
	place_4th_result_df.drop([1], axis=1, inplace=True)

	place_result_df = pd.merge(place_1st_result_df, place_2nd_result_df, left_index=True, right_index=True)
	place_result_df = pd.merge(place_result_df, place_3rd_result_df, left_index=True, right_index=True)
	place_result_df = pd.merge(place_result_df, place_4th_result_df, left_index=True, right_index=True)
	place_result_df['same_place_flg'] = place_result_df['place_4th_number'].apply(lambda x : 1 if x != '0' else 0)

	place_result_df.drop(['place_4th_number', 'place_4th_reward'], axis=1, inplace=True)

	place_result_df['place_1st_number'] = place_result_df['place_1st_number'].apply(lambda x : 0 if x == 'nan' else x)
	place_result_df['place_1st_reward'] = place_result_df['place_1st_reward'].apply(lambda x : 0 if x == None else x)

	place_result_df['place_1st_number'] = place_result_df['place_1st_number'].astype('int')
	place_result_df['place_2nd_number'] = place_result_df['place_2nd_number'].astype('int')
	place_result_df['place_3rd_number'] = place_result_df['place_3rd_number'].astype('int')
	place_result_df['place_1st_reward'] = place_result_df['place_1st_reward'].astype('float')
	place_result_df['place_2nd_reward'] = place_result_df['place_2nd_reward'].astype('float')
	place_result_df['place_3rd_reward'] = place_result_df['place_3rd_reward'].astype('float')

	#馬連配当から払い戻しの情報を抽出するために加工する。
	quinella_str_list = []
	for i in range(len(df)):
		quinella_str_list.append(repr(df['馬連配当'][i]))

	#馬連配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	quinella_str_list_df = pd.DataFrame(quinella_str_list)
	quinella_str_list_df = quinella_str_list_df.rename(columns={0:'quinella_result'})
	quinella_str_list_df['quinella_result'] = quinella_str_list_df['quinella_result'].str.replace("'","")
	quinella_str_list_df['quinella_result'] = quinella_str_list_df['quinella_result'].str.split('\\')
	quinella_result_list = list(quinella_str_list_df['quinella_result'])
	quinella_result_temp_df = pd.DataFrame(quinella_result_list)

	quinella_1_2_df = pd.DataFrame(list(quinella_result_temp_df[0].str.split('-')))
	quinella_1_2_df = quinella_1_2_df.rename(columns={0:'quinella_1', 1:'quinella_2'})

	quinella_reward_df = pd.DataFrame(quinella_result_temp_df[2].str.replace('\([0-9]*\)',''))
	quinella_reward_df[2] = quinella_reward_df[2].str.strip()
	quinella_reward_df = pd.DataFrame(list(quinella_reward_df[2].str.split('/')))
	quinella_reward_df = quinella_reward_df.rename(columns={0:'quinella_reward'})
	quinella_reward_df['same_quinella_flg'] = quinella_reward_df[1].apply(lambda x : 0 if x == None else 1)
	quinella_reward_df.drop([1], axis=1, inplace=True)

	quinella_result_df = pd.merge(quinella_1_2_df, quinella_reward_df, left_index=True, right_index=True)
	quinella_result_df['quinella_1'] = quinella_result_df['quinella_1'].astype('int')
	quinella_result_df['quinella_2'] = quinella_result_df['quinella_2'].astype('int')
	quinella_result_df['quinella_reward'] = quinella_result_df['quinella_reward'].astype('float')

	#馬単配当から払い戻しの情報を抽出するために加工する。
	exacta_str_list = []
	for i in range(len(df)):
		exacta_str_list.append(repr(df['馬単配当'][i]))

	#馬単配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	exacta_str_list_df = pd.DataFrame(exacta_str_list)
	exacta_str_list_df = exacta_str_list_df.rename(columns={0:'exacta_result'})
	exacta_str_list_df['exacta_result'] = exacta_str_list_df['exacta_result'].str.replace("'","")
	exacta_str_list_df['exacta_result'] = exacta_str_list_df['exacta_result'].str.split('\\')
	exacta_result_list = list(exacta_str_list_df['exacta_result'])
	exacta_result_temp_df = pd.DataFrame(exacta_result_list)

	exacta_1_2_df = pd.DataFrame(list(exacta_result_temp_df[0].str.split('-')))
	exacta_1_2_df = exacta_1_2_df.rename(columns={0:'exacta_1st', 1:'exacta_2nd'})

	exacta_reward_df = pd.DataFrame(exacta_result_temp_df[2].str.replace('\([0-9]*\)',''))
	exacta_reward_df[2] = exacta_reward_df[2].str.strip()
	exacta_reward_df = pd.DataFrame(list(exacta_reward_df[2].str.split('/')))
	exacta_reward_df = exacta_reward_df.rename(columns={0:'exacta_reward'})
	exacta_reward_df['same_exacta_flg'] = exacta_reward_df[1].apply(lambda x : 0 if x == None else 1)
	exacta_reward_df.drop([1], axis=1, inplace=True)

	exacta_result_df = pd.merge(exacta_1_2_df, exacta_reward_df, left_index=True, right_index=True)
	exacta_result_df['exacta_1st'] = exacta_result_df['exacta_1st'].astype('int')
	exacta_result_df['exacta_2nd'] = exacta_result_df['exacta_2nd'].astype('int')
	exacta_result_df['exacta_reward'] = exacta_result_df['exacta_reward'].astype('float')

	#ワイド配当から払い戻しの情報を抽出するために加工する。
	quinella_place_str_list = []
	for i in range(len(df)):
		quinella_place_str_list.append(repr(df['ワイド配当'][i]))

	#ワイド配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	quinella_place_str_list_df = pd.DataFrame(quinella_place_str_list)
	quinella_place_str_list_df[0] = quinella_place_str_list_df[0].str.replace("'","")
	quinella_place_str_list_df[0] = quinella_place_str_list_df[0].str.split('\\')
	quinella_place_list = list(quinella_place_str_list_df[0])
	quinella_place_result_temp_df = pd.DataFrame(quinella_place_list)

	quinella_place_1st_1_2 = pd.DataFrame(list(quinella_place_result_temp_df[0].str.split('-')))
	quinella_place_1st_1_2 = quinella_place_1st_1_2.rename(columns={0:'quinella_place_1st_1', 1:'quinella_place_1st_2'})
	quinella_place_1st_reward = pd.DataFrame(list(quinella_place_result_temp_df[2].str.split('/')))
	quinella_place_1st_reward.drop([1], axis=1, inplace=True)
	quinella_place_1st_reward = quinella_place_1st_reward.rename(columns={0:'quinella_place_1st_reward'})
	quinella_place_1st_reward['quinella_place_1st_reward'] = quinella_place_1st_reward['quinella_place_1st_reward'].str.replace('\([0-9]*\)','')

	quinella_place_2nd_1_2 = pd.DataFrame(list(quinella_place_result_temp_df[2].str.split('/')))
	quinella_place_2nd_1_2.drop([0], axis=1, inplace=True)
	quinella_place_2nd_1_2 = pd.DataFrame(list(quinella_place_2nd_1_2[1].str.split('-')))
	quinella_place_2nd_1_2 = quinella_place_2nd_1_2.rename(columns={0:'quinella_place_2nd_1', 1:'quinella_place_2nd_2'})
	quinella_place_2nd_reward = pd.DataFrame(list(quinella_place_result_temp_df[4].str.split('/')))
	quinella_place_2nd_reward.drop([1], axis=1, inplace=True)
	quinella_place_2nd_reward = quinella_place_2nd_reward.rename(columns={0:'quinella_place_2nd_reward'})
	quinella_place_2nd_reward['quinella_place_2nd_reward'] = quinella_place_2nd_reward['quinella_place_2nd_reward'].str.replace('\([0-9]*\)','')

	quinella_place_3rd_1_2 = pd.DataFrame(list(quinella_place_result_temp_df[4].str.split('/')))
	quinella_place_3rd_1_2.drop([0], axis=1, inplace=True)
	quinella_place_3rd_1_2 = pd.DataFrame(list(quinella_place_3rd_1_2[1].str.split('-')))
	quinella_place_3rd_1_2 = quinella_place_3rd_1_2.rename(columns={0:'quinella_place_3rd_1', 1:'quinella_place_3rd_2'})
	quinella_place_3rd_reward = pd.DataFrame(list(quinella_place_result_temp_df[6].str.split('/')))
	quinella_place_3rd_reward.drop([1], axis=1, inplace=True)
	quinella_place_3rd_reward = quinella_place_3rd_reward.rename(columns={0:'quinella_place_3rd_reward'})
	quinella_place_3rd_reward['quinella_place_3rd_reward'] = quinella_place_3rd_reward['quinella_place_3rd_reward'].str.replace('\([0-9]*\)','')

	same_quinella_place_flg_df = pd.DataFrame(list(quinella_place_result_temp_df[6].str.split('/')))
	same_quinella_place_flg_df['same_quinella_place_flg'] = same_quinella_place_flg_df[1].apply(lambda x : 0 if x == None else 1)
	same_quinella_place_flg_df.drop([0,1], axis=1, inplace=True)

	quinella_place_result_df = pd.merge(quinella_place_1st_1_2, quinella_place_1st_reward, left_index=True, right_index=True)
	quinella_place_result_df = pd.merge(quinella_place_result_df, quinella_place_2nd_1_2, left_index=True, right_index=True)
	quinella_place_result_df = pd.merge(quinella_place_result_df, quinella_place_2nd_reward, left_index=True, right_index=True)
	quinella_place_result_df = pd.merge(quinella_place_result_df, quinella_place_3rd_1_2, left_index=True, right_index=True)
	quinella_place_result_df = pd.merge(quinella_place_result_df, quinella_place_3rd_reward, left_index=True, right_index=True)
	quinella_place_result_df = pd.merge(quinella_place_result_df, same_quinella_place_flg_df, left_index=True, right_index=True)

	quinella_place_result_df['quinella_place_1st_1'] = quinella_place_result_df['quinella_place_1st_1'].astype('int') 
	quinella_place_result_df['quinella_place_1st_2'] = quinella_place_result_df['quinella_place_1st_2'].astype('int')
	quinella_place_result_df['quinella_place_1st_reward'] = quinella_place_result_df['quinella_place_1st_reward'].astype('float')
	quinella_place_result_df['quinella_place_2nd_1'] = quinella_place_result_df['quinella_place_2nd_1'].astype('int') 
	quinella_place_result_df['quinella_place_2nd_2'] = quinella_place_result_df['quinella_place_2nd_2'].astype('int')
	quinella_place_result_df['quinella_place_2nd_reward'] = quinella_place_result_df['quinella_place_2nd_reward'].astype('float')
	quinella_place_result_df['quinella_place_3rd_1'] = quinella_place_result_df['quinella_place_3rd_1'].astype('int') 
	quinella_place_result_df['quinella_place_3rd_2'] = quinella_place_result_df['quinella_place_3rd_2'].astype('int')
	quinella_place_result_df['quinella_place_3rd_reward'] = quinella_place_result_df['quinella_place_3rd_reward'].astype('float')

	#3連複配当から払い戻しの情報を抽出するために加工する。
	trio_str_list = []
	for i in range(len(df)):
		trio_str_list.append(repr(df['3連複配当'][i]))

	#3連複配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	trio_str_list_df = pd.DataFrame(trio_str_list)
	trio_str_list_df[0] = trio_str_list_df[0].str.replace("'","")
	trio_str_list_df[0] = trio_str_list_df[0].str.split('\\')
	trio_list = list(trio_str_list_df[0])
	trio_result_temp_df = pd.DataFrame(trio_list)

	trio_1_2_3 = pd.DataFrame(list(trio_result_temp_df[0].str.split('-')))
	trio_1_2_3 = trio_1_2_3.rename(columns={0:'trio_1', 1:'trio_2', 2:'trio_3'})
	trio_reward = pd.DataFrame(list(trio_result_temp_df[2].str.split('/')))
	trio_reward[0] = trio_reward[0].str.replace('\([0-9]*','')
	trio_reward = trio_reward.rename(columns={0:'trio_reward'})
	trio_reward.drop([1,2], axis=1, inplace=True)

	same_trio_flg_df = pd.DataFrame(list(trio_result_temp_df[2].str.split('/')))
	same_trio_flg_df['same_trio_flg'] = same_trio_flg_df[2].apply(lambda x : 0 if x == None else 1)
	same_trio_flg_df.drop([0,1,2], axis=1, inplace=True)

	trio_result_df = pd.merge(trio_1_2_3, trio_reward, left_index=True, right_index=True)
	trio_result_df = pd.merge(trio_result_df, same_trio_flg_df, left_index=True, right_index=True)

	trio_result_df['trio_1'] = trio_result_df['trio_1'].astype('int')
	trio_result_df['trio_2'] = trio_result_df['trio_2'].astype('int')
	trio_result_df['trio_3'] = trio_result_df['trio_3'].astype('int')
	trio_result_df['trio_reward'] = trio_result_df['trio_reward'].astype('float')

	#3連単配当から払い戻しの情報を抽出するために加工する。
	trifecta_str_list = []
	for i in range(len(df)):
		trifecta_str_list.append(repr(df['3連単配当'][i]))

	#3連単配当の情報の分割と同着決着判定フラグを作成する。
	#同着決着レースは学習データから外すため。
	trifecta_str_list_df = pd.DataFrame(trifecta_str_list)
	trifecta_str_list_df[0] = trifecta_str_list_df[0].str.replace("'","")
	trifecta_str_list_df[0] = trifecta_str_list_df[0].str.split('\\')
	trifecta_list = list(trifecta_str_list_df[0])
	trifecta_result_temp_df = pd.DataFrame(trifecta_list)

	trifecta_1_2_3 = pd.DataFrame(list(trifecta_result_temp_df[0].str.split('-')))
	trifecta_1_2_3 = trifecta_1_2_3.rename(columns={0:'trifecta_1', 1:'trifecta_2', 2:'trifecta_3'})
	trifecta_reward = pd.DataFrame(list(trifecta_result_temp_df[2].str.split('/')))
	trifecta_reward[0] = trifecta_reward[0].str.replace('\([0-9]*','')
	trifecta_reward = trifecta_reward.rename(columns={0:'trifecta_reward'})
	trifecta_reward.drop([1,2], axis=1, inplace=True)

	same_trifecta_flg_df = pd.DataFrame(list(trifecta_result_temp_df[2].str.split('/')))
	same_trifecta_flg_df['same_trifecta_flg'] = same_trifecta_flg_df[2].apply(lambda x : 0 if x == None else 1)
	same_trifecta_flg_df.drop([0,1,2], axis=1, inplace=True)

	trifecta_result_df = pd.merge(trifecta_1_2_3, trifecta_reward, left_index=True, right_index=True)
	trifecta_result_df = pd.merge(trifecta_result_df, same_trifecta_flg_df, left_index=True, right_index=True)

	trifecta_result_df['trifecta_1'] = trifecta_result_df['trifecta_1'].astype('int')
	trifecta_result_df['trifecta_2'] = trifecta_result_df['trifecta_2'].astype('int')
	trifecta_result_df['trifecta_3'] = trifecta_result_df['trifecta_3'].astype('int')
	trifecta_result_df['trifecta_reward'] = trifecta_result_df['trifecta_reward'].astype('float')

	#払い戻し情報を統合する。
	refund_df = pd.merge(win_result_df, place_result_df, left_index=True, right_index=True)
	refund_df = pd.merge(refund_df, quinella_result_df, left_index=True, right_index=True)
	refund_df = pd.merge(refund_df, exacta_result_df, left_index=True, right_index=True)
	refund_df = pd.merge(refund_df, quinella_place_result_df, left_index=True, right_index=True)
	refund_df = pd.merge(refund_df, trio_result_df, left_index=True, right_index=True)
	refund_df = pd.merge(refund_df, trifecta_result_df, left_index=True, right_index=True)
	df_key = df[['年','月','日','場所','レース番号']]
	refund_df = pd.merge(df_key, refund_df, left_index=True, right_index=True)
	
	#払い戻し情報から目的変数を作成する。
	#ここが肝になる。
	refund_df['1st_reward'] = (
		refund_df['win_reward'] 
		+ refund_df['place_1st_reward'] 
		+ (refund_df['quinella_reward'] / 2)
		#+ (refund_df['exacta_reward'] * 0.7)
		+ (refund_df['quinella_place_1st_reward'] / 2)
		+ (refund_df['quinella_place_2nd_reward'] / 2)
		+ (refund_df['trio_reward'] / 3)
		+ (refund_df['trifecta_reward'] * 0.5)
	)*1.5

	refund_df['2nd_reward'] = (
		refund_df['place_2nd_reward']
		+ (refund_df['quinella_reward'] / 2)
		#+ (refund_df['exacta_reward'] * 0.3)
		+ (refund_df['quinella_place_1st_reward'] / 2)
		+ (refund_df['quinella_place_3rd_reward'] / 2)
		+ (refund_df['trio_reward'] / 3)
		+ (refund_df['trifecta_reward'] * 0.3)
	)*1.0

	refund_df['3rd_reward'] = (
		refund_df['place_3rd_reward']
		+ (refund_df['quinella_place_2nd_reward'] / 2)
		+ (refund_df['quinella_place_3rd_reward'] / 2)
		+ (refund_df['trio_reward'] / 3)
		+ (refund_df['trifecta_reward'] * 0.2)
	)*0.5

	reward_1_df = refund_df[['年','月','日','場所','レース番号','place_1st_number', '1st_reward']]
	reward_1_df = reward_1_df.rename(columns={'place_1st_number':'馬番', '1st_reward':'target'})

	reward_2_df = refund_df[['年','月','日','場所','レース番号','place_2nd_number', '2nd_reward']]
	reward_2_df = reward_2_df.rename(columns={'place_2nd_number':'馬番', '2nd_reward':'target'})

	reward_3_df = refund_df[['年','月','日','場所','レース番号','place_3rd_number', '3rd_reward']]
	reward_3_df = reward_3_df.rename(columns={'place_3rd_number':'馬番', '3rd_reward':'target'})

	reward_df = pd.concat([reward_1_df, reward_2_df, reward_3_df], axis=0)
	reward_df = reward_df.reset_index()
	reward_df.drop('index', axis=1, inplace=True)

	#同着レースフラグを作成し、レースの同着有無を判定するデータを作成する。
	refund_df['same_race_flg'] = (
		refund_df['same_win_flg']
		+ refund_df['same_place_flg']
		+ refund_df['same_quinella_flg']
		+ refund_df['same_exacta_flg']
		+ refund_df['same_quinella_place_flg']
		+ refund_df['same_trio_flg']
		+ refund_df['same_trifecta_flg']
	)

	refund_df['same_race_flg'] = refund_df['same_race_flg'].apply(lambda x : 0 if x == 0 else 1)

	return refund_df, reward_df