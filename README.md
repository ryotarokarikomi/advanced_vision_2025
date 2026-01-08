# アドバンスドビジョン - 課題
Kerasを用いたオートエンコーダ

## 概要
本リポジトリでは，オートエンコーダを用いて，MNIST手書き数字データセットに対して画像の次元圧縮および再構成をおこなった．

オートエンコーダは以下の2つのネットワークから構成されるニューラルネットワークである．
  - エンコーダ：入力データを低次元の潜在表現に写像
  - デコーダ：潜在表現から元の入力データを再構成

本実装では，全結合層を重ねた多層構造を採用し，
誤差逆伝播法によって入力と出力の再構成誤差を最小化するように学習を行った．

## 実行のためのセットアップ
### リポジトリのクローン
```
git clone https://github.com/ryotarokarikomi/advanced_vision_2025.git
cd advanced_vision_2025
```
### 仮想環境の作成と有効化
```
python -m venv .venv
source .venv/bin/activate
```
### 依存ライブラリのインストール
```
pip install -r requirements.txt
```

## 実行方法
### デフォルト設定での実行
```
python src/main.py
```
**デフォルトのパラメータ**
- エポック数：100
- バッチサイズ：256
- 潜在表現の次元数：32
- 結果の保存先：`results/`
### パラメータを変更して実行
例）エポック数を20，潜在表現の次元数を16
```
python src/main.py --epochs 20 --encoding_dim 16
```



## 参考文献
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
