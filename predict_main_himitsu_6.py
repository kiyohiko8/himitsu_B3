"""
##プログラム概要
- 修正版評価手法B
- 被験者に知っているひみつ道具を入力させる
	+ 質問形式（ランダム）
	+ 入力した結果から被験者の知識リストを作成	
- 学習済みモデルに入力
	+ 既知と推測される語をリスト化	
- 推定結果を出力
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import himitsu_data_gd_6
import collected_himitsu_data_6
import collected_himitsu_sort
import numpy as np
import os
import random
import gc




"""被験者に知識情報を入力させる（修正前）
def mk_user_know(sorted):

	#知識情報入力
	wiselist = []
	cnt = 0

	for item in sorted:
		print(item[1])
		data = input("これを知っていれば1, 知らなければ0を入力してください:")
		#例外処理
		while 0 < 1:
			if data == str(0):
				break
			elif data == str(1):
				break
			else:
				data = input("入力しなおしてください:")
		#入力データリストに追加
		if data == str(1):
			wiselist.append(item[1])
			cnt += 1
			#知っている知識が3個以上になったらループを抜け出す(2018/1/12変更)
			if cnt > 2:
				break

	return wiselist
"""

	
"""被験者に知識情報を入力させる（修正後）"""
def mk_user_know(himitsu):

	#指定道具数数ランダム抽出			
	a = random.sample(himitsu, 100)

	#知識情報入力
	wiselist = []
	cnt = 0
	for item in a:
		print(item)
		print("これを知っていれば1, 知らなければ0を入力してください")
		data = input()
		while 0 < 1:
			if data == str(0):
				break
			elif data == str(1):
				break
			else:
				data = input("入力しなおしてください")
				
		#入力データリストに追加
		if data == str(1):
			wiselist.append(item)
			cnt += 1
			#知っている知識が3個以上になったらループを抜け出す
			if cnt > 2:
				break


		
	return wiselist



	


"""被験者の回答から入力用データを作成"""
def mk_input_data(wiselist, all_word_list):
	input_data = []
	u_data = np.zeros(271, int)
		
	for i, word in enumerate(all_word_list):
		for item in wiselist:
			if item == word:
				u_data[i] = 1
				input_data.append(u_data)
				break

		
					
					
	return input_data



"""結果の元となる既知情報辞書の作成"""
def mk_know_dic(x, y, vec):
	know_dic  = {}		
	for (j, data) in enumerate(x):
		i = j + 1
		if y[0][j] >= 0.4:
			for k,v in vec.items():
				if vec[k] == i:
					know_dic[k] = y[0][j]
					
	return know_dic
	





if __name__ == "__main__":

	#全ひみつ道具データの読み込み
	himitsu  = himitsu_data_gd_6.mk_allword_list()
	#ひみつ道具ベクトルの作成
	word_vec = himitsu_data_gd_6.mk_vec(himitsu)
	#収集データの読み込み
	collected = collected_himitsu_data_6.read_csv("himitsu_data.csv")
	#出現順にsortしたデータの読み込み
	sorted   = collected_himitsu_sort.count_sort(collected, himitsu)
	
	#学習結果の読み込み：中のファイル名は学習済みのモデルおよびパラメータ
	model = model_from_json(open('predict_model_himitsu_1.json').read())
	model.load_weights('predict_weights_himitsu_1.h5')
	
	#概要の出力
	model.summary();
	model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1), metrics=['accuracy'])

	
	while 0 < 1 :
		print("これは手法Bによる推定器です")
		#ユーザ入力部
		wise_list = mk_user_know(himitsu)#修正後の入力法
		#wise_list = mk_user_know(sorted)#修正前の入力法

		print(wise_list)
		#推定器への入力用データの作成
		input_data = mk_input_data(wise_list, himitsu)

		input_data = np.array(input_data)
		print(input_data)
	
		#ユーザデータの推定値出力...入力は被験者の入力作業によって得られたデータ
		#predintionsは出力であり、それぞれの単語の推定値を確率で出力
		predictions = model.predict(input_data, 1, 1)
		print(predictions)


		#既知・未知推定情報の取得
		know_dic = mk_know_dic(himitsu, predictions, word_vec)
			
			
		#結果の出力
		print("\n\nあなたが知ってそうなひみつ道具は......\n\n")
		

		for k, v in know_dic.items():
			print("[", k, "既知確率:", str(round(v*100, 2)), "%" "]")
		
		print("\n\n\n")
		
		print("システムを終了しますか？")
		a = input("終了するならyキーを, 続けるならnキーを押してください:")
		if a == "y":
			print("\n\n\nシステム作動を終了します...\n\n\n")
			break
			
		elif a == "n":
			print("\n\n\nシステム作動を継続します...\n\n\n")
			

			
	gc.collect()
		
		
		
		

	

