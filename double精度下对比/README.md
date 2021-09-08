# double精度下对比

## 发现可能的问题：
（1）在double精度下，pytorch的`matmul_qk / np.sqrt(dk)`与paddle的`matmul_qk / np.sqrt(dk)`结果不一致。

# 这样误差很大。
```python
import numpy as np
import torch
import paddle
paddle.set_device("cpu")
inputs = np.random.normal(size=(10,1024)).astype("double")
scale = np.sqrt(80)

def compare(a, b):
    a = a.numpy()
    b = b.numpy()
    print("mean dif:", np.abs(a - b).mean())
    print("max dif:", np.abs(a - b).max())

compare(torch.from_numpy(inputs) / scale, paddle.to_tensor(inputs,dtype=paddle.float64) / scale)
mean dif: 1.0744490034772033e-09
max dif: 5.539800673748374e-09
```

# 必须这样误差才能为0
```python
import numpy as np
import torch
import paddle
paddle.set_device("cpu")
inputs = np.random.normal(size=(10,1024)).astype("double")
scale = np.sqrt(80)

def compare(a, b):
    a = a.numpy()
    b = b.numpy()
    print("mean dif:", np.abs(a - b).mean())
    print("max dif:", np.abs(a - b).max())

compare(torch.from_numpy(inputs) / scale, paddle.to_tensor(inputs,dtype=paddle.float64) / paddle.to_tensor(scale,dtype=paddle.float64))
mean dif: 0.0
max dif: 0.0
```

（2）positional_encoding构建出来的结果误差也很大，实际测试的时候。

# 这个误差看起来还很正常，但是进入模型的时候误差不正常。
```python
import paddle
import paddle.nn as nn
import torch
import numpy as np

paddle.set_default_dtype("float64")
class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        out.stop_gradient = True
        position_ids = paddle.arange(
            0, n_pos, dtype=out.dtype).unsqueeze(1)
        indices = paddle.arange(
            0, dim // 2, dtype=out.dtype).unsqueeze(0)

        indices = 10000.0 **(-2 * indices / dim)
        embeddings = paddle.matmul(position_ids, indices)
        sentinel = dim // 2
        out[:, 0:sentinel] = paddle.sin(embeddings)
        out[:, sentinel:] = paddle.cos(embeddings)

        return out

    @paddle.no_grad()
    def forward(self, position_ids):
        return super().forward(position_ids)

def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size, dtype=torch.double):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(
        torch.arange(position, dtype=dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
        d_model_size,
    )

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

paddleencoding = SinusoidalPositionalEmbedding(512,64)
torchencoding = positional_encoding(512,64)
def compare(a, b):
    a = a.numpy()
    b = b.numpy()
    print("mean dif:", np.abs(a - b).mean())
    print("max dif:", np.abs(a - b).max())

compare(torchencoding[range(500)],paddleencoding(paddle.arange(500)))
mean dif: 6.367781409825424e-16
max dif: 2.842170943040401e-14
```

# 于是我修改了下模型中的SinusoidalPositionalEmbedding,使用torch先计算出来positional_encoding，然后给paddle的模型赋值。
```python
import torch    
def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates

def positional_encoding(position, d_model_size, dtype):
    dtype = torch.float64
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(
        torch.arange(position, dtype=dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
        d_model_size,
    )

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        out.stop_gradient = True
        s = positional_encoding(n_pos,dim,torch.float64)
        out[:,:] = paddle.to_tensor(s.numpy(),dtype=paddle.get_default_dtype())[:,:]
        
        return out

    @paddle.no_grad()
    def forward(self, position_ids):
        return super().forward(position_ids)
```

## 修改完上述2处后，进行double精度下对齐，本次在GPU条件下对齐48层的模型（设备V100 32G）。

(1) 下载权重 https://huggingface.co/ctrl/ 中的`pytorch_model.bin`放进 `hg/ctrl`文件夹对应目录。
(2) 运行`python convert.py`将这个权重转化为paddle的双精度的权重。
(3) 修改`compare_lm.py` Line21的`pd = True`变量，然后运行`python compare_lm.py`,生成paddle版本的`logits`和`hidden states`结果。
(4) 修改`compare_lm.py` Line21的`pd = False`变量，然后运行`python compare_lm.py`,生成pytorch版本的`logits`和`hidden states`结果。
(5) 运行`python bijiao.py`，得到中间变量的结果`bijiaojieguo.txt`。
(6) 打开`compare_loss_hiddenstate.ipynb`，一行一行运行，对比`logits`和`hidden states`结果。

### 结果如下：
#### logits对比
```python
mean dif: 1.2361560583031692e-14
max dif: 7.096545573404001e-13
```

#### hidden_states对比
```python
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.544227481675603e-15
max dif: 2.1316282072803006e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 6.372488843075394e-15
max dif: 2.2737367544323206e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 7.704470604299839e-15
max dif: 2.8421709430404007e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.0433180292915443e-14
max dif: 2.8421709430404007e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.3795053377647449e-14
max dif: 1.3642420526593924e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.8718442442079318e-14
max dif: 2.5011104298755527e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.3590409226838297e-14
max dif: 3.183231456205249e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.8800644183030957e-14
max dif: 3.751665644813329e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.484849375243685e-14
max dif: 3.751665644813329e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.349350347861801e-14
max dif: 3.865352482534945e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.0786044141212454e-14
max dif: 3.979039320256561e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.770044950734535e-14
max dif: 4.092726157978177e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 6.698958494592813e-14
max dif: 4.092726157978177e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 7.477340530026523e-14
max dif: 4.092726157978177e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 8.497180413794592e-14
max dif: 4.320099833421409e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 9.773695313023009e-14
max dif: 4.149569576838985e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.1280460659552429e-13
max dif: 4.916511642250043e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.278453473888645e-13
max dif: 5.233147248873138e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.433327236577975e-13
max dif: 6.0254023992456496e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.5887813796782585e-13
max dif: 7.048583938740194e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.7130930459539495e-13
max dif: 7.219114195322618e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.8405062799612281e-13
max dif: 8.753886504564434e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.9961959592072206e-13
max dif: 9.43600753089413e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.113312394580512e-13
max dif: 9.322320693172514e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.243641700286155e-13
max dif: 9.43600753089413e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.359342366674923e-13
max dif: 1.000444171950221e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.473834696518719e-13
max dif: 1.0686562745831907e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.5682251455905863e-13
max dif: 1.2050804798491299e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.692909434947604e-13
max dif: 1.2846612662542611e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.816392715653435e-13
max dif: 1.4438228390645236e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.865356232533957e-13
max dif: 1.48929757415317e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.9829135021812406e-13
max dif: 1.5802470443304628e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.104504953010587e-13
max dif: 1.750777300912887e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.183413774541431e-13
max dif: 1.7962520360015333e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.293205138549448e-13
max dif: 1.8417267710901797e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.381371260770228e-13
max dif: 1.9326762412674725e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.4640976362891827e-13
max dif: 1.9554136088117957e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.5583608692709436e-13
max dif: 2.0236257114447653e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.711588068308722e-13
max dif: 2.0463630789890885e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.7875945562215127e-13
max dif: 2.1032064978498966e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.899684491198e-13
max dif: 2.1827872842550278e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.032774660772911e-13
max dif: 2.262368070660159e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.194529949730455e-13
max dif: 2.3078428057488054e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.315290652773871e-13
max dif: 2.4783730623312294e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.4989300777625514e-13
max dif: 2.5693225325085223e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.586745658045088e-13
max dif: 2.717115421546623e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.742571622351195e-13
max dif: 2.751221472863108e-11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.089596511437409e-15
max dif: 1.290079154614432e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```