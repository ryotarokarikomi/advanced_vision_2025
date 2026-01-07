import numpy as np

import keras
from keras import layers
from keras.datasets import mnist


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


# ==============================
# 2) オートエンコーダ定義
# ==============================
encoding_dim = 32  # 潜在次元（圧縮後の表現）

# 入力（MNIST: 784次元ベクトル）
input_img = keras.Input(shape=(784,), name="input_img")

# Encoder: 784 -> 32
encoded = layers.Dense(encoding_dim, activation="relu", name="encoded")(input_img)

# Decoder: 32 -> 784
decoded = layers.Dense(784, activation="sigmoid", name="decoded")(encoded)

# Autoencoder: 入力 -> 再構成
autoencoder = keras.Model(inputs=input_img, outputs=decoded, name="autoencoder")


# ==============================
# 3) Encoder / Decoder を分離
# ==============================
# Encoder: 入力 -> 潜在表現
encoder = keras.Model(inputs=input_img, outputs=encoded, name="encoder")

# Decoder: 潜在表現 -> 再構成
encoded_input = keras.Input(shape=(encoding_dim,), name="encoded_input")

# autoencoderの最後の層（Dense(784, sigmoid)）を取り出して再利用
decoder_layer = autoencoder.layers[-1]
decoder_output = decoder_layer(encoded_input)

decoder = keras.Model(inputs=encoded_input, outputs=decoder_output, name="decoder")


# ==============================
# 4) 学習設定
# ==============================
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# ==============================
# 5) 学習
# ==============================
autoencoder.fit(x_train, x_train,
  epochs=50,
  batch_size=256,
  shuffle=True,
  validation_data=(x_test, x_test)
)
