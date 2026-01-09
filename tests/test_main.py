import unittest
import numpy as np

from src.main import models

# テストクラスの定義
class TestAutoEncoder(unittest.TestCase):
  def test_forward_shape(self):
    autoencoder, _, _ = models(32) # モデルの生成
    x = np.random.rand(4, 784).astype("float32") # ダミー入力の生成
    y = autoencoder.predict(x, verbose=0)
    self.assertEqual(y.shape, x.shape)

if __name__ == "__main__":
  unittest.main()
