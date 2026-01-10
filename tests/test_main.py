import unittest
import numpy as np

from src.main import models

# テストクラスの定義
class TestAutoEncoder(unittest.TestCase):
  def test_shape(self):
    encoding_dim = 32
    batch_size = 4
    input_dim = 784
    
    # モデルの生成
    autoencoder, encoder, decoder = models(encoding_dim)

    # ダミー入力の生成
    x = np.random.rand(batch_size, input_dim).astype("float32")

    # 1）オートエンコーダ全体の出力チェック (入力と同じ 784)
    y = autoencoder.predict(x, verbose=0)
    self.assertEqual(y.shape, (batch_size, input_dim), "Autoencoder output shape is wrong")

    # 2）エンコーダの出力チェック (潜在空間 32)
    z = encoder.predict(x, verbose=0)
    self.assertEqual(z.shape, (batch_size, encoding_dim), "Encoder output shape is wrong")

    # 3）デコーダの出力チェック (元の 784 に戻るか)
    x_recon = decoder.predict(z, verbose=0) # 変数名を合わせる
    self.assertEqual(x_recon.shape, (batch_size, input_dim), "Decoder output shape is wrong") # input_dimと比較

if __name__ == "__main__":
  unittest.main()
