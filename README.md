# 混雑状況のリアルタイム可視化AI
## コンセプト
コロナ下の感染防止のための混雑回避

・定期的に撮影した画像から人を認識し
   混雑状況を解析

・結果をクラウドと連携し可視化

・解析には小型PCのRaspberry Piを使用

・ローカルで画像撮影/分析/結果出力し、MQTT通信（高速、高セキュリティ）でクラウド（aws）にデータを送信、データ保管/可視化を行う

## ポイント

・定期的に自動でカメラ撮影するよう組み込む

・撮影した画像解析はリアルタイム性が求められる（数分程度）


・混雑状況の定義（混雑率）

>>混雑率[人/m2] = 人数 ÷ 使用場所の広さ（面積）

・離れた場所から情報取得 / 情報セキュリティの確保
![IoT jpeg 001](https://user-images.githubusercontent.com/62229682/89100320-1dffa280-d431-11ea-87ea-6dc75d9a7151.jpeg)
## 動作方法

① evaluate.pyを実行

② カメラ使用環境の広さ[人/m2]を入力する。

③ enterを押し、開始

④ 以降は、カメラ撮影及びその画像に対し物体検出モデル（Yolo）が呼び出され、時間、人数、混雑率を出力される。

⑤ 出力結果は自動でaws IoT Coreに送信される（事前にaws IoT CoreとのMQTT通信できるよう設定、各種証明書やキーを取得する必要あり）

## エッジデバイス〜awsの接続設定
aws IoT coredでモノ証明書、プライベートキー、エンドポイントを取得
certフォルダに格納し、evaluate.pyのP29〜32に記入。

## エッジデバイス（ラズパイ）の環境

tensorflow : 1.14.0

keras : 2.1.4
