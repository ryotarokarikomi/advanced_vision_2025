import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.datasets import mnist

# ==============================
# 0) コマンドライン引数の設定
# ==============================
parser = argparse.ArgumentParser(description='Autoencoder MNIST Training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--encoding_dim', type=int, default=32, help='Dimension of the latent space')
parser.add_argument('--out', type=str, default='results/', help='Directory to save results')

args = parser.parse_args()

# ==============================
# 1) データ読み込み & 前処理
# ==============================
(x_train, _), (x_test, _) = mnist.load_data()

# 0-255 -> 0-1 に正規化（sigmoid出力 & BCE loss と相性が良い）
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 28x28 -> 784 に flatten（Dense入力に合わせる）
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("x_train:", x_train.shape)  # (60000, 784)
print("x_test :", x_test.shape)   # (10000, 784)

# 結果の保存先ディレクトリ
RESULT_DIR = args.out
os.makedirs(RESULT_DIR, exist_ok=True)


# ==============================
# 2) オートエンコーダ定義
# ==============================

encoding_dim = args.encoding_dim

# 入力（MNIST: 784次元ベクトル）
input_img = keras.Input(shape=(784,), name="input_img")

# Encoder: 784 -> 256 -> 128 -> 64 -> 32
encoded = layers.Dense(256, activation='relu')(input_img)
encoded = layers.Dense(128, activation='relu')(encoded)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder: 32 -> 64 -> 128 -> 256 -> 784
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(256, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

autoencoder = keras.Model(inputs=input_img, outputs=decoded, name="autoencoder")

# ==============================
# 3) Encoder / Decoder を分離
# ==============================

# Encoder: 入力 -> 潜在表現
encoder = keras.Model(inputs=input_img, outputs=encoded, name="encoder")

# Decoder: 潜在表現 -> 再構成
encoded_input = keras.Input(shape=(encoding_dim,), name="encoded_input")

# autoencoderの後半3つの層を順番に適用
x = autoencoder.layers[-4](encoded_input)
x = autoencoder.layers[-3](x)
x = autoencoder.layers[-2](x)
decoder_output = autoencoder.layers[-1](x)

decoder = keras.Model(inputs=encoded_input, outputs=decoder_output, name="decoder")


# ==============================
# 4) 学習設定
# ==============================
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# ==============================
# 5) 学習
# ==============================
history = autoencoder.fit(x_train, x_train,
  epochs=args.epochs,
  batch_size=args.batch_size,
  shuffle=True,
  validation_data=(x_test, x_test)
)

# ==============================
# 6) 結果の保存
# ==============================

# 再構成画像
decoded_images = autoencoder.predict(x_test[:10], verbose="0")

plt.figure(figsize=(20, 4))
for i in range(10):
  ax = plt.subplot(2, 10, i + 1)
  plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
  plt.axis("off")

  ax = plt.subplot(2, 10, i + 11)
  plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
  plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "reconstruction.png"), dpi=200)
plt.close()

# 学習曲線（損失）
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "loss_curve.png"), dpi=200)
plt.close()
