#ライブラリをインポートする。
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

import copy
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import linear_model
import pickle

def lgbmtrain(df):
	#obeject型変数をcategory型に変換する。		
	df_object_col_list = df.select_dtypes(include=['object'])
	for var in df_object_col_list:
		df[var] = df[var].astype('category')

	#説明変数のデータフレームを作成する。
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

	train_explanatory_var_df = df[explanatory_var]
	train_explanatory_var_df = train_explanatory_var_df.reset_index()
	train_explanatory_var_df.drop('index', axis=1, inplace=True)

	#目的変数のデータフレームを作成する。
	train_target_var_df = df[['target']]

	#lgb用に学習データを変換する。	
	lgb_train = lgb.Dataset(train_explanatory_var_df, train_target_var_df)

	#lgb用のパラメーターを設定する。
	lgbm_params = {
		'objective': 'regression',
		'metric': 'rmse',
		'num_boost_round': 100,
		#'early_stopping_round': 50, 
		'max_depth': 6,
		'max_leaf': 40,
		'boosting': 'gbdt'
	}

	#lgbで学習する。
	lgb_model = lgb.train(lgbm_params, lgb_train, num_boost_round=50)

	return lgb_model

def multiregtrain(df):

	#説明変数のデータフレームを作成する。
	explanatory_var = ['人気', '単勝オッズ', '父馬名単勝フラグ_調整平均', '父馬名複勝フラグ_調整平均', '父馬名単勝払戻金_調整平均',
	'父馬名複勝払戻金_調整平均', '母の父馬名単勝フラグ_調整平均', '母の父馬名複勝フラグ_調整平均',
	'母の父馬名単勝払戻金_調整平均', '母の父馬名複勝払戻金_調整平均', '父タイプ単勝フラグ_調整平均',
	'父タイプ複勝フラグ_調整平均', '父タイプ単勝払戻金_調整平均', '父タイプ複勝払戻金_調整平均', '母父タイプ単勝フラグ_調整平均',
	'母父タイプ複勝フラグ_調整平均', '母父タイプ単勝払戻金_調整平均', '母父タイプ複勝払戻金_調整平均', '騎手単勝フラグ_調整平均',
	'騎手複勝フラグ_調整平均', '騎手単勝払戻金_調整平均', '騎手複勝払戻金_調整平均', '調教師単勝フラグ_調整平均',
	'調教師複勝フラグ_調整平均', '調教師単勝払戻金_調整平均', '調教師複勝払戻金_調整平均', '馬主単勝フラグ_調整平均',
	'馬主複勝フラグ_調整平均', '馬主単勝払戻金_調整平均', '馬主複勝払戻金_調整平均', '生産者単勝フラグ_調整平均',
	'生産者複勝フラグ_調整平均', '生産者単勝払戻金_調整平均', '生産者複勝払戻金_調整平均']

	train_explanatory_var_df = df[explanatory_var]
	train_explanatory_var_df = train_explanatory_var_df.reset_index()
	train_explanatory_var_df.drop('index', axis=1, inplace=True)
	train_explanatory_var_df = train_explanatory_var_df.fillna(0)
	
	#目的変数のデータフレームを作成する。
	train_target_var_df = df[['target']]

	#重回帰で学習する。
	mreg = linear_model.LinearRegression()
	multireg_model = mreg.fit(train_explanatory_var_df, train_target_var_df)

	return multireg_model
