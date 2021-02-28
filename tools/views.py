#ライブラリをインポートする。
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

import copy
import csv
import glob
import re
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os.path

#自作関数をインポートする。		
from tools.process import makedatamart, makemodel, makeresult

#ファイルを出力するファイルパスの変数を設定する。
FILE_OUT_DIR = "./tools/static/tools/media/"

def index(request):

	DEFAULT_DIR = os.getcwd()
	os.chdir(FILE_OUT_DIR)

	csv_file_lst = glob.glob("*.csv")
	if len(csv_file_lst) > 0:
		for csvlist in csv_file_lst:
			if csvlist != "analytics_df.csv" and csvlist != "train_df.csv" and csvlist != "multireg_mst.csv":
				fs = FileSystemStorage()
				fs.delete(csvlist)
	
	os.chdir(DEFAULT_DIR)

	return render(request, 'tools/index.html')

def train(request):

	#html画面から3ファイルが選択⇨ボタンが押下されると以下の処理を実行する。	
	if request.method == 'POST' and request.FILES['racecard'] and request.FILES['raceinfo'] and request.FILES['training']:

		###html画面から選択された3ファイルからデータマート作成⇨学習をする。
		
		#各ファイルのファイル名を変数化する。
		myfile1 = request.FILES['racecard']
		myfile2 = request.FILES['raceinfo']
		myfile3 = request.FILES['training']

		#各ファイルをdjangoのファイルを扱う領域に移動する。
		fs = FileSystemStorage()

		#出馬表ファイルをdjangoのファイルを扱う領域にコピーする。
		racecard_fn_tmp = fs.save(myfile1.name, myfile1)
		#racecard_fn = "http://localhost:8000/static/tools/media/" + racecard_fn_tmp
		racecard_fn = FILE_OUT_DIR + racecard_fn_tmp

		#レース結果ファイルをdjangoのファイルを扱う領域にコピーする。
		raceinfo_fn_tmp = fs.save(myfile2.name, myfile2)
		#raceinfo_fn = "http://localhost:8000/static/tools/media/" + raceinfo_fn_tmp
		raceinfo_fn = FILE_OUT_DIR + raceinfo_fn_tmp

		#新馬時調教ファイルをdjangoのファイルを扱う領域にコピーする。
		training_fn_tmp = fs.save(myfile3.name, myfile3)
		#training_fn = "http://localhost:8000/static/tools/media/" + training_fn_tmp
		training_fn = FILE_OUT_DIR + training_fn_tmp
		
		#勾配ブースティング学習用データマートを作成する。
		train_df = makedatamart.makelgbmtraindf(racecard_fn, raceinfo_fn, training_fn)
		#勾配ブースティングモデルを作成し、djangoのファイルを扱う領域に保存する。
		train_df = train_df[train_df['クラス名'] == '新馬']
		#train_df = train_df[train_df['クラス名'] != 'None']
		lgb_model = makemodel.lgbmtrain(train_df)

		train_df.to_csv(FILE_OUT_DIR + "train_df.csv", index=False)
		pickle.dump(lgb_model, open(FILE_OUT_DIR + "lgb_model.sav", 'wb'))

		del lgb_model

		#重回帰用	学習用データマートを作成する。
		train_df = makedatamart.makemultiregtraindf(racecard_fn, raceinfo_fn, training_fn)
		train_df.to_csv(FILE_OUT_DIR + "multireg_mst.csv", index=False)

		#重回帰モデルを作成し、djangoのファイルを扱う領域に保存する。
		train_df = train_df[train_df['クラス名'] == '新馬']
		#train_df = train_df[train_df['クラス名'] != 'None']
		multireg_model = makemodel.multiregtrain(train_df)
		pickle.dump(multireg_model, open(FILE_OUT_DIR + "multireg_model.sav", 'wb'))
		
		del multireg_model

		#過去結果分析用に分析用データをanalytics_df.csvの名前で出力する。
		analytics_df = makedatamart.makeanalyticsdf(racecard_fn, raceinfo_fn)
		analytics_df.to_csv(FILE_OUT_DIR + "analytics_df.csv", index=False)

		#不要になったファイルを削除する。
		fs.delete(racecard_fn_tmp)
		fs.delete(raceinfo_fn_tmp)
		fs.delete(training_fn_tmp)

		#predictのhtmlサイトへ推移する。
		return redirect('/predict')

	#Topページのhtmlサイトをレンダーする。
	return render(request, 'tools/train.html')

def predict(request):

	#html画面から2ファイルが選択⇨ボタンが押下されると以下の処理を実行する。
	if request.method == 'POST' and request.FILES['racecard'] and request.FILES['training']:

		###html画面から選択された2ファイルから予想をする。
		
		#各ファイルのファイル名を変数化する。
		myfile1 = request.FILES['racecard']
		myfile2 = request.FILES['training']
		
		#各ファイルをdjangoのファイルを扱う領域に移動する。
		fs = FileSystemStorage()

		#出馬表ファイルをdjangoのファイルを扱う領域にコピーする。
		racecard_fn_tmp = fs.save(myfile1.name, myfile1)
		racecard_fn = FILE_OUT_DIR + racecard_fn_tmp

		#新馬時調教ファイルをdjangoのファイルを扱う領域にコピーする。
		training_fn_tmp = fs.save(myfile2.name, myfile2)
		training_fn = FILE_OUT_DIR + training_fn_tmp

		#過去結果分析用に出馬表ファイルをracecard.csvの名前で出力する。
		racecard_df = pd.read_csv(racecard_fn, encoding='SJIS')
		racecard_df.to_csv(FILE_OUT_DIR + "racecard.csv", index=False)

		#lgbm予想用データマートを作成する。
		predict_df1 = makedatamart.makelgbmpredictdf(racecard_fn, training_fn)

		#lgbm予想結果を出力する。
		predict_result_df1, predict_detail_df1, feature_importance_df1 = makeresult.lgbmpredict(predict_df1)

		#重回帰予想用データマートを作成する。
		multireg_traindf = pd.read_csv(FILE_OUT_DIR + "multireg_mst.csv")
		predict_df2 = makedatamart.makemultiregpredictdf(racecard_fn, training_fn, multireg_traindf)

		#重回帰予想結果を出力する。
		predict_result_df2 = makeresult.multiregpredict(predict_df2)

		#グラフ出力用に予想結果を加工する。
		graph_result_df, graph_bar_result_df, graph_predict_detail_df, graph_result_detail_df, graph_result_feature_importance_df \
		= makeresult.makeresultgraphdf(predict_df1, predict_result_df1, predict_result_df2, predict_detail_df1, feature_importance_df1)

		#グラフ出力用のCSVファイルをdjangoのファイルを扱う領域に作成する。
		graph_result_df.to_csv(FILE_OUT_DIR + "graph_result.csv", index=False)
		graph_bar_result_df.to_csv(FILE_OUT_DIR + "graph_bar_result.csv", index=False)
		graph_predict_detail_df.to_csv(FILE_OUT_DIR + "graph_predict_result.csv", index=False)
		graph_result_detail_df.to_csv(FILE_OUT_DIR + "graph_result_detail.csv", index=False)
		graph_result_feature_importance_df.to_csv(FILE_OUT_DIR + "graph_result_feature_importance.csv", index=False)

		#不要になったファイルを削除する。
		#fs.delete(racecard_fn_tmp)
		fs.delete(training_fn_tmp)

		#resultのhtmlサイトへ推移する。
		return redirect('/result')

	return render(request, 'tools/predict.html')

def result(request):
	racecond = pd.read_csv(FILE_OUT_DIR + "racecard.csv")
	racecond = racecond.rename(columns={'芝・ダート':'芝ダ', '馬場状態(暫定)':'馬場状態'})
	racecond['馬場状態'] = racecond['馬場状態'].apply(lambda x : '良' if x == '良(暫定)' else x)
	racecond['馬場状態'] = racecond['馬場状態'].apply(lambda x : '稍' if x == '稍(暫定)' else x)
	racecond['馬場状態'] = racecond['馬場状態'].apply(lambda x : '重' if x == '重(暫定)' else x)
	racecond['馬場状態'] = racecond['馬場状態'].apply(lambda x : '不' if x == '不(暫定)' else x)
	if racecond['月'][0] < 10:
		if racecond['日'][0] < 10:
			racecond1 = str(racecond['年'][0]) + "/" + "0" + str(racecond['月'][0]) + "/" + "0" + str(racecond['日'][0])
		else:
			racecond1 = str(racecond['年'][0]) + "/" + "0" + str(racecond['月'][0]) + "/" + str(racecond['日'][0])
	else:
		if racecond['日'][0] < 10:
			racecond1 = str(racecond['年'][0]) + "/" + str(racecond['月'][0]) + "/" + "0" + str(racecond['日'][0])
		else:
			racecond1 = str(racecond['年'][0]) + "/" + str(racecond['月'][0]) + "/" + str(racecond['日'][0])

	racecond2 = racecond['場所'][0] + str(racecond['R'][0]) + "R" + "  " + racecond['クラス名'][0]
	racecond3 = racecond['芝ダ'][0] + str(racecond['距離'][0]) + "  " + racecond['馬場状態'][0]

	racecond = racecond[['場所', '芝ダ', '距離', 'クラス名', '馬場状態']][:1]
	racecond.to_csv(FILE_OUT_DIR + "racecond.csv", index=False)

	return render(request, 'tools/result.html', {'racecond1': racecond1, 'racecond2': racecond2, 'racecond3': racecond3})

def analytics(request):
	
	if os.path.isfile(FILE_OUT_DIR + "analytics_done_df.csv") is False:
		#同一セッション内でanalytics.htmlを1度通った事をに明示するためにダミーファイルを出力する。	
		#index.htmlに戻ったら、セッションが切れたと判定している。
		analytics_done = [[0],[1]]
		analytics_done_df = pd.DataFrame(analytics_done)
		analytics_done_df.to_csv(FILE_OUT_DIR + "analytics_done_df.csv", index=False)

		#勝率、複勝率、単勝回収率、複勝回収率集計用にデータを作成する。

		#分析用データをデータフレーム化する。(train関数(makedatamartのmaketraindf関数)でanalytics_df.csvを作成している。)		
		analytics_df = pd.read_csv(FILE_OUT_DIR + "analytics_df.csv")
		#出馬表ファイルをデータフレーム化する。(predict関数でracecard.csvを作成している。)
		racecard_df = pd.read_csv(FILE_OUT_DIR + "racecard.csv")

		#集計結果描画用のCSVファイルを出力する関数を実行する。
		makeresult.makeanalyticsgraphdf(analytics_df, racecard_df)

	return render(request, 'tools/analytics.html')
