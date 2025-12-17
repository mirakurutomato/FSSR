# Fast Single-Scale Retinex (FSSR) Implementation

## 概要 (Overview)

本リポジトリは、低照度環境下での画像認識精度向上を目的とした **Fast Single-Scale Retinex (FSSR)** アルゴリズムのPython実装です。
従来のSingle-Scale Retinex (SSR) をベースに、輝度成分（Y）のみへの処理適用と分離可能フィルタ（Separable Filter）によるガウシアンブラーの高速化を行い、リアルタイム性を確保しています。

本コードでは、低照度環境を想定した入力映像に対してFSSRによる強調処理を行い、**MediaPipe Hands** を用いた手部骨格検出の精度比較（補正前 vs 補正後）をリアルタイムで実行します。

## 特徴 (Features)

* **高速なRetinex処理**: 輝度成分のみの処理とSeparable Gaussian Blurにより計算コストを低減。
* **リアルタイム比較**: Webカメラ映像に対し、補正前と補正後の骨格検出結果を並列表示。
* **パラメータ調整**: 実行中にSigma（ブラーの強度）を動的に調整可能。
* **定量評価**: フレームごとの処理時間（Sep/Full）、FPS、およびMSE（平均二乗誤差）を算出。

## 参考文献 (Reference)

本実装は以下の研究に基づいています。

* **論文**: Fast Single-Scale Retinexの提案と低照度環境におけるリアルタイム骨格検出性能の評価
* **著者**: 奥河 董馬 (Toma Okugawa) - 弓削商船高等専門学校
* **DOI**: [10.51094/jxiv.1897](https://doi.org/10.51094/jxiv.1897)

## 環境構築 (Installation)

動作には Python 3.x が必要です。以下のコマンドで必要なライブラリをインストールしてください。

```bash
pip install -r requirements.txt

```

## 使用方法 (Usage)

以下のコマンドでスクリプトを実行します。Webカメラが接続されている必要があります。

```bash
python main.py

```

### 操作方法 (Controls)

実行中に以下のキーボード操作が可能です。

* `Esc`: アプリケーションを終了します。
* `s`: 現在のフレーム（比較画像）をスクリーンショットとして保存します。
* 画像は自動生成される `screenshots/` フォルダに保存されます。

* `+` / `=`: Sigma値を増加させます（ブラー強度アップ）。
* `-` / `_`: Sigma値を減少させます（ブラー強度ダウン）。

## ファイル構成 (File Structure)

```
.
├── screenshots/       # スクリーンショット保存用ディレクトリ（自動生成）
├── LICENSE            # ライセンス
├── main.py            # FSSRアルゴリズムおよびベンチマーク用メインスクリプト
├── README.md          # 本ドキュメント
└── requirements.txt   # 依存ライブラリ一覧

```

## Author

奥河 董馬 (Toma Okugawa)

弓削商船高等専門学校 (National Institute of Technology, Yuge College)
