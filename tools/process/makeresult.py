#ライブラリをインポートする。
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

import copy
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

#ファイルを出力するファイルパスの変数を設定する。
FILE_OUT_DIR = "./tools/static/tools/media/"

def lgbmpredict(df):

	df['馬体重'] = df['馬体重'].astype('float')
	df['斤量'] = df['斤量'].astype('float')
	df['斤量差'] = df['斤量差'].astype('float')
	#df['馬番'] = df['馬番'].astype('category') 
	df['人気'] = df['人気'].astype('float')
	#df['距離'] = df['距離'].astype('category')
	df['コーナー回数'] = df['コーナー回数'].astype('int')
	#df['コーナー回数'] = df['コーナー回数'].astype('category')
	df['コース区分'] = df['コース区分'].astype('category') 

	#説明変数のリストを作成する。
	explanatory_var = ['場所', '性別', '父馬名', '母の父馬名', '父タイプ', '母父タイプ',
					   '馬体重', '騎手', '調教師', '所属', 
					   '斤量', '斤量差', '馬番', '人気', '単勝オッズ', '単勝オッズ_std',
					   '芝ダ', '距離', '非根幹フラグ', '頭数', 'コーナー回数', 'コース区分', '天候', '馬場状態',
					   'Time1_max', 'Time1_min', 'Time1_mean', 'Time1_count',
					   'Time2_max', 'Time2_min', 'Time2_mean', 'Time2_count',
					   'Time3_max', 'Time3_min', 'Time3_mean', 'Time3_count',
					   'Time4_max', 'Time4_min', 'Time4_mean', 'Time4_count',
					   'Lap1_max', 'Lap1_min', 'Lap1_mean', 'Lap1_count',
					   'Lap2_max', 'Lap2_min', 'Lap2_mean', 'Lap2_count',
					   'Lap3_max', 'Lap3_min', 'Lap3_mean', 'Lap3_count',
					   'Lap4_max', 'Lap4_min', 'Lap4_mean', 'Lap4_count']

	#説明変数データを作成する。
	train_explanatory_var_df = df[explanatory_var]
	train_explanatory_var_df = train_explanatory_var_df.reset_index()
	train_explanatory_var_df.drop('index', axis=1, inplace=True)

	#モデルを読み込む。
	with open(FILE_OUT_DIR + "lgb_model.sav", 'rb') as lgb_model:
		lgb_model = pickle.load(lgb_model)

	#通常モードで予想する。
	predict = lgb_model.predict(train_explanatory_var_df)
	predict_result_df = pd.DataFrame(predict)

	#レーダーチャート用に回帰式出力モードで予想する。
	predict_detail = lgb_model.predict(train_explanatory_var_df, pred_contrib=True)
	#predict_detail = lgb_model.predict(train_explanatory_var_df, pred_leaf=True)
	predict_detail_df = pd.DataFrame(predict_detail)

	#変数重要度データを作成する。
	feature_importance_df = pd.DataFrame()
	feature_importance_df['Feature'] = lgb_model.feature_name()
	# feature_importance_df['importance'] = lgb_model.feature_importance()
	feature_importance_df['importance'] = lgb_model.feature_importance(importance_type='gain')

	return predict_result_df, predict_detail_df, feature_importance_df

def multiregpredict(df):

	#説明変数のリストを作成する。
	explanatory_var = ['人気', '単勝オッズ', '父馬名単勝フラグ_調整平均', '父馬名複勝フラグ_調整平均', '父馬名単勝払戻金_調整平均',
	'父馬名複勝払戻金_調整平均', '母の父馬名単勝フラグ_調整平均', '母の父馬名複勝フラグ_調整平均',
	'母の父馬名単勝払戻金_調整平均', '母の父馬名複勝払戻金_調整平均', '父タイプ単勝フラグ_調整平均',
	'父タイプ複勝フラグ_調整平均', '父タイプ単勝払戻金_調整平均', '父タイプ複勝払戻金_調整平均', '母父タイプ単勝フラグ_調整平均',
	'母父タイプ複勝フラグ_調整平均', '母父タイプ単勝払戻金_調整平均', '母父タイプ複勝払戻金_調整平均', '騎手単勝フラグ_調整平均',
	'騎手複勝フラグ_調整平均', '騎手単勝払戻金_調整平均', '騎手複勝払戻金_調整平均', '調教師単勝フラグ_調整平均',
	'調教師複勝フラグ_調整平均', '調教師単勝払戻金_調整平均', '調教師複勝払戻金_調整平均', '馬主単勝フラグ_調整平均',
	'馬主複勝フラグ_調整平均', '馬主単勝払戻金_調整平均', '馬主複勝払戻金_調整平均', '生産者単勝フラグ_調整平均',
	'生産者複勝フラグ_調整平均', '生産者単勝払戻金_調整平均', '生産者複勝払戻金_調整平均']

	#説明変数データを作成する。
	train_explanatory_var_df = df[explanatory_var]
	train_explanatory_var_df = train_explanatory_var_df.reset_index()
	train_explanatory_var_df.drop('index', axis=1, inplace=True)

	#モデルを読み込む。
	with open(FILE_OUT_DIR + "multireg_model.sav", 'rb') as multireg_model:
		multireg_model = pickle.load(multireg_model)

	#通常モードで予想する。
	predict = multireg_model.predict(train_explanatory_var_df)
	predict_result_df = pd.DataFrame(predict)

	return predict_result_df

def makeresultgraphdf(predict_df, predict_result_df1, predict_result_df2, predict_detail_df, feature_importance_df):

	#出馬表の結果を加工する。
	result = pd.merge(predict_df, predict_result_df1, left_index=True, right_index=True)
	result = result.rename(columns={0:'メイン予想結果'})
	result = pd.merge(result, predict_result_df2, left_index=True, right_index=True)
	result = result.rename(columns={0:'サブ予想結果'})
	result['メイン予想結果'] = result['メイン予想結果'].astype('int')
	result['サブ予想結果'] = result['サブ予想結果'].astype('int')
	result = result[['年','月','日','場所', 'レース番号', '馬番', '馬名', '父馬名', '母の父馬名', '性別', '年齢', '騎手', '斤量', '調教師', '所属', '馬体重','人気', '単勝オッズ', 'メイン予想結果', 'サブ予想結果']]
	graph_result_df = result[['馬番', '馬名', '父馬名', '母の父馬名','性別', '年齢', '騎手', '斤量', '調教師', '所属', '馬体重', '人気', '単勝オッズ', 'メイン予想結果', 'サブ予想結果']]

	#説明変数のリストを作成する。
	explanatory_var = ['場所', '性別', '父馬名', '母の父馬名', '父タイプ', '母父タイプ',
					   '馬体重', '騎手', '調教師', '所属',
					   '斤量', '斤量差', '馬番', '人気', '単勝オッズ', '単勝オッズ_std',
					   '芝ダ', '距離', '非根幹フラグ', '頭数', 'コーナー回数', 'コース区分', '天候', '馬場状態',
					   'Time1_max', 'Time1_min', 'Time1_mean', 'Time1_count',
					   'Time2_max', 'Time2_min', 'Time2_mean', 'Time2_count',
					   'Time3_max', 'Time3_min', 'Time3_mean', 'Time3_count',
					   'Time4_max', 'Time4_min', 'Time4_mean', 'Time4_count',
					   'Lap1_max', 'Lap1_min', 'Lap1_mean', 'Lap1_count',
					   'Lap2_max', 'Lap2_min', 'Lap2_mean', 'Lap2_count',                  
					   'Lap3_max', 'Lap3_min', 'Lap3_mean', 'Lap3_count',                  
					   'Lap4_max', 'Lap4_min', 'Lap4_mean', 'Lap4_count']

	#説明変数の係数のデータフレーム を作成する。
	explanatory_var.append('切片')
	predict_detail_df.columns = explanatory_var

	graph_result_df['メイン予想結果'] = graph_result_df['メイン予想結果'] - predict_detail_df['切片']
	graph_result_df['メイン予想結果'] = graph_result_df['メイン予想結果'].astype('int')

	graph_result_df['lgbm予想順位'] = graph_result_df['メイン予想結果'].rank(ascending=False)
	graph_result_df['メインAI印'] = graph_result_df['lgbm予想順位'].apply(lambda x : " " if x >= 4 else x)
	graph_result_df['メインAI印'] = graph_result_df['メインAI印'].apply(lambda x : "◉" if x == 1 else x)
	graph_result_df['メインAI印'] = graph_result_df['メインAI印'].apply(lambda x : "○" if x == 2 else x)
	graph_result_df['メインAI印'] = graph_result_df['メインAI印'].apply(lambda x : "▲" if x == 3 else x)

	graph_result_df['サブ予想結果'] = graph_result_df['サブ予想結果'].astype('int')

	graph_result_df['multireg予想順位'] = graph_result_df['サブ予想結果'].rank(ascending=False)
	graph_result_df['サブAI印'] = graph_result_df['multireg予想順位'].apply(lambda x : " " if x >= 4 else x)
	graph_result_df['サブAI印'] = graph_result_df['サブAI印'].apply(lambda x : "◉" if x == 1 else x)
	graph_result_df['サブAI印'] = graph_result_df['サブAI印'].apply(lambda x : "○" if x == 2 else x)
	graph_result_df['サブAI印'] = graph_result_df['サブAI印'].apply(lambda x : "▲" if x == 3 else x)

	graph_result_df.drop(['メイン予想結果','lgbm予想順位','サブ予想結果','multireg予想順位'], axis=1, inplace=True)
	graph_bar_result_df = result[['年','月','日','場所', 'レース番号', '馬番', '馬名', 'メイン予想結果', 'サブ予想結果']]

	#レーダーチャートで確認したい項目に絞る。
	result_detail_df = predict_detail_df[['場所', '性別', '父馬名', '母の父馬名', '父タイプ', '母父タイプ',
					   '馬体重', '騎手', '調教師', '所属', '斤量', '斤量差', '馬番', '人気', 
					   '芝ダ', '距離', '非根幹フラグ', '頭数', 'コーナー回数', '天候', '馬場状態', 'Lap4_max', 'Lap4_min']]
	horse_name = predict_df[['馬名']]
	graph_predict_detail_df = pd.merge(horse_name, predict_detail_df, left_index=True, right_index=True)
	graph_result_detail_df = pd.merge(horse_name, result_detail_df, left_index=True, right_index=True)

	#変数重要度の高い順番にソートする。
	graph_result_feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

	return graph_result_df, graph_bar_result_df, graph_predict_detail_df, graph_result_detail_df, graph_result_feature_importance_df

def makeanalyticsgraphdf(analytics_df, racecard_df):

	#過去の結果と出馬表情報を結合することで、出馬する馬の適性を確認する。

	#過去の結果を別名のデータフレームにする。
	tmp_analytics_df = analytics_df.copy()

	#出馬表情報を過去の結果と結合するために、加工する。
	tmp_racecard_df = racecard_df.rename(columns={'父':'父馬名', '母父':'母の父馬名', '番':'馬番','芝・ダート':'芝ダ', '馬場状態(暫定)':'馬場状態'})
	tmp_racecard_df['馬場状態'] = tmp_racecard_df['馬場状態'].str.replace('(','')
	tmp_racecard_df['馬場状態'] = tmp_racecard_df['馬場状態'].str.replace(')','')
	tmp_racecard_df['馬場状態'] = tmp_racecard_df['馬場状態'].str.replace('暫定','')

	#馬毎に異なる変数*馬毎で同一の集計対象ごとに勝率、複勝率、単勝回収率、複勝回収率および適性ランクを算出する。
	def make_count_df(analytics_df, racecard_df, var1, var2):
		nodupdfname = racecard_df.drop_duplicates(subset=[var1])[[var1]]
		gb_df = pd.merge(analytics_df, nodupdfname, on=var1, how='inner')
		gb_df['カウントフラグ'] = 1
		gb_df['単勝フラグ'] = 0
		gb_df['単勝払戻金'] = 0
		gb_df.loc[(gb_df['馬番'] == gb_df['win_number']), '単勝フラグ'] = 100
		gb_df.loc[(gb_df['馬番'] == gb_df['win_number']), '単勝払戻金'] = gb_df['win_reward']    
		gb_df['複勝フラグ'] = 0
		gb_df['複勝払戻金'] = 0
		gb_df.loc[(gb_df['馬番'] == gb_df['place_1st_number']), '複勝フラグ'] = 100
		gb_df.loc[(gb_df['馬番'] == gb_df['place_2nd_number']), '複勝フラグ'] = 100
		gb_df.loc[(gb_df['馬番'] == gb_df['place_3rd_number']), '複勝フラグ'] = 100
		gb_df.loc[(gb_df['馬番'] == gb_df['place_1st_number']), '複勝払戻金'] = gb_df['place_1st_reward']
		gb_df.loc[(gb_df['馬番'] == gb_df['place_2nd_number']), '複勝払戻金'] = gb_df['place_2nd_reward']
		gb_df.loc[(gb_df['馬番'] == gb_df['place_3rd_number']), '複勝払戻金'] = gb_df['place_3rd_reward']
		gb_df = gb_df.groupby([var1, var2]).mean()[['単勝フラグ', '複勝フラグ', '単勝払戻金', '複勝払戻金']].reset_index()

		gb_rank_df = gb_df.copy()
		gb_rank_df[var1 + '_期待スコア'] = gb_rank_df['単勝フラグ'] + gb_rank_df['複勝フラグ'] + gb_rank_df['単勝払戻金'] + gb_rank_df['複勝払戻金']
		gb_rank_df[var1 + var2 + '_rank'] = gb_rank_df.groupby([var2])[var1 + '_期待スコア'].rank(ascending=False)
		gb_rank_df = gb_rank_df[[var1, var2, var1 + var2 + '_rank']]
		racecard_rank_df = pd.merge(racecard_df, gb_rank_df, on=[var1, var2], how='left')

		return gb_df, racecard_rank_df

	#馬毎に異なる集計対象の変数のリストを作成する。
	var1_list = ['父馬名', '母の父馬名', '騎手', '調教師']
	#馬毎で同一の集計対象の変数のリストを作成する。
	var2_list = ['クラス名', '芝ダ', '距離', '場所', '馬場状態', '頭数']

	#馬毎に異なる変数*馬毎で同一の集計対象ごとに勝率、複勝率、単勝回収率、複勝回収率を算出し、CSV出力する。
	for var2 in var2_list:
		if var2 == 'クラス名':
			fname2 = 'class'
		if var2 == '芝ダ':
			fname2 = 'shibada'
		if var2 == '距離':
			fname2 = 'kyori'
		if var2 == '場所':
			fname2 = 'basho'
		if var2 == '馬場状態':
			fname2 = 'baba'
		if var2 == '頭数':
			fname2 = 'tosu'

		for var1 in var1_list:
			gb_df, _ = make_count_df(tmp_analytics_df, tmp_racecard_df, var1, var2)
			if var1 == '父馬名':
				fname1 = 'shuboba'
			if var1 == '母の父馬名':
				fname1 = 'hahachichi'
			if var1 == '騎手':
				fname1 = 'kisyu'
			if var1 == '調教師':
				fname1 = 'tyokyoshi'

			gb_df.to_csv(FILE_OUT_DIR + fname1 + '_' + fname2 + '_gb.csv', index=False)

	#馬毎に異なる変数*馬毎で同一の集計対象ごとに適性ランクを算出し、CSV出力する。
	i = 0
	for var2 in var2_list:
		rank_var_list = []
		for var1 in var1_list:
			if i == 0:
				_, racecard_rank_df = make_count_df(tmp_analytics_df, tmp_racecard_df, var1, var2)
				i = i + 1
			else:
				_, racecard_rank_df = make_count_df(tmp_analytics_df, racecard_rank_df, var1, var2)

			rank_var = var1 + var2 + '_rank'
			rank_var_list.append(rank_var)

		j = 0
		for rank_var in rank_var_list:
			if j == 0:
				racecard_rank_df[var2 + '適性'] = racecard_rank_df[rank_var]
				j = j + 1
			else:
				racecard_rank_df[var2 + '適性'] = racecard_rank_df[var2 + '適性'] + racecard_rank_df[rank_var]

			del racecard_rank_df[rank_var]
	
		racecard_rank_df[var2 + '適性'] = np.floor(racecard_rank_df[var2 + '適性'].rank())
	
	#必要な変数を選択し、CSV出力する。
	racecard_rank_df = racecard_rank_df[['馬番','  馬名','父馬名','母の父馬名','騎手','調教師','人気',' 単勝','クラス名適性','芝ダ適性','距離適性','場所適性','馬場状態適性','頭数適性']]
	racecard_rank_df.rename(columns={'  馬名':'馬名', ' 単勝':'単勝オッズ', 'クラス名適性':'新馬戦適性', '馬場状態適性':'馬場適性'}, inplace=True)
	racecard_rank_df.to_csv(FILE_OUT_DIR + 'racecard_rank_df.csv', index=False)
