# 分析

由于ctrl模型权重大小6个G，因此本次测试都是在cpu环境下进行测试。

# 总结：
```python
# 随机初始化的48层的CTRL模型（误差正常，说明模型搭建没有错误）：
compare loss:
mean difference: tensor(9.5367e-07)
max difference: tensor(9.5367e-07)
compare logits:
mean difference: tensor(5.6924e-07)
max difference: tensor(5.2452e-06)
compare 最后一层hidden_states:
mean difference: tensor(7.6511e-07)
max difference: tensor(5.7220e-06)

# 加载48层CTRL的预训练权重（误差很大，模型误差与权重初始化相关）：
compare loss:
mean difference: tensor(9.5367e-07)
max difference: tensor(9.5367e-07)
compare logits:
mean difference: tensor(4.7009e-06)
max difference: tensor(0.0003)
compare 最后一层hidden_states:
mean difference: tensor(1.2011e-06)
max difference: tensor(5.8532e-05)
```
# 分析可能的原因：（都是CPU环境下）
- （1）ctrl采用了pre layer norm、而bert等模型采用了post layer norm。前者是不通过layernorm的hidden_state的比较，后者是通过layernorm的hidden_state的比较。通常来说通过layer norm后误差会变小。
- （2）https://github.com/PaddlePaddle/Paddle/issues/35118 在layer norm在cpu环境下，相同的输入，相同的初始化的条件下进行误差比较，发现最大误差就已经达到10-6级别。
- （3）https://github.com/PaddlePaddle/Paddle/issues/35123 在linear在cpu环境下，相同的输入，相同的初始化（初始化的分布方差可能很大）的条件下进行误差比较，发现最大误差很大10-4级别。
- （4）`CTRL+带有mask的结果.txt`文件第73-77行，是`attention output`（最大误差1.7583370208740234e-06）通过`FFN`得到的结果（最大误差0.00014495849609375），这个是通过linear线性层得到的结果，这个误差突然变大的状况与原因（3）有点类似，因此我猜测误差大的原因是（3）导致的。 
- （5）`CTRL+带有mask的结果.txt`文件第25-29行，对query（最大误差6.9141387939453125e-06）和key（最大误差6.198883056640625e-06）进行矩阵乘法得到的结果（最大误差0.000396728515625），该结果误差也是非常大。同样感觉是导致误差大的原因。
- （6）`CTRL+带有mask的结果.txt`文件第41-45行，给缩放后的`logits，scaled_attention_logits`添加了`语言模型的MASK`，主要操作是加上了很大的负数，原先最大误差（4.57763671875e-05），加上负数后最大误差（0.0009765625）。按照常理来说加上负数后的最大误差应该小于原先最大误差。不过这里没啥影响，因为最后这个结果还会通过softmax，通过softmax后最大误差（3.6954879760742188e-06）。


# 比较方法
```bash
# （1）生成随机初始化的48层CTRL
python generate_random.py
# （2）删除jilu文件夹并重新创建
rm jilu -r
mkdir jilu
# （3）修改compare_lm.py的21行和23行，将值改为 random_ctrl。然后进行比较。
python compare_lm.py
# （4）运行完毕，可以比较已经保存的中间变量,最终结果会保存为`bijiaojieguo.txt`。
python bijiao.py 
```

```bash
# （1）将paddle和pytorch版本的预训练权重放进original_ctrl文件夹。
# （2）删除jilu文件夹并重新创建
rm jilu -r
mkdir jilu
# （3）修改compare_lm.py的21行和23行，将值改为 original_ctrl。然后进行比较。
python compare_lm.py
# （4）运行完毕，可以比较已经保存的中间变量,最终结果会保存为`bijiaojieguo.txt`。
python bijiao.py 
```

## 这里是我已经运行得到的结果。
## 随机初始化的CTRL，带有mask。
```python
compare loss:
mean difference: tensor(9.5367e-07)
max difference: tensor(9.5367e-07)
compare logits:
mean difference: tensor(5.6924e-07)
max difference: tensor(5.2452e-06)
compare hidden_states:
mean difference: tensor(1.5762e-07)
max difference: tensor(7.6294e-06)
mean difference: tensor(6.6285e-07)
max difference: tensor(1.0610e-05)
mean difference: tensor(9.2678e-07)
max difference: tensor(1.2398e-05)
mean difference: tensor(1.1340e-06)
max difference: tensor(1.4663e-05)
mean difference: tensor(1.3219e-06)
max difference: tensor(1.5259e-05)
mean difference: tensor(1.4898e-06)
max difference: tensor(1.5497e-05)
mean difference: tensor(1.6433e-06)
max difference: tensor(1.5259e-05)
mean difference: tensor(1.7900e-06)
max difference: tensor(1.6689e-05)
mean difference: tensor(1.9287e-06)
max difference: tensor(1.6689e-05)
mean difference: tensor(2.0612e-06)
max difference: tensor(1.7881e-05)
mean difference: tensor(2.1893e-06)
max difference: tensor(1.8775e-05)
mean difference: tensor(2.3062e-06)
max difference: tensor(1.8835e-05)
mean difference: tensor(2.4191e-06)
max difference: tensor(1.8597e-05)
mean difference: tensor(2.5299e-06)
max difference: tensor(1.9550e-05)
mean difference: tensor(2.6389e-06)
max difference: tensor(2.0504e-05)
mean difference: tensor(2.7422e-06)
max difference: tensor(2.0742e-05)
mean difference: tensor(2.8450e-06)
max difference: tensor(2.0981e-05)
mean difference: tensor(2.9511e-06)
max difference: tensor(2.2113e-05)
mean difference: tensor(3.0487e-06)
max difference: tensor(2.2054e-05)
mean difference: tensor(3.1430e-06)
max difference: tensor(2.2650e-05)
mean difference: tensor(3.2357e-06)
max difference: tensor(2.3842e-05)
mean difference: tensor(3.3282e-06)
max difference: tensor(2.4557e-05)
mean difference: tensor(3.4160e-06)
max difference: tensor(2.4796e-05)
mean difference: tensor(3.5009e-06)
max difference: tensor(2.4796e-05)
mean difference: tensor(3.5883e-06)
max difference: tensor(2.5749e-05)
mean difference: tensor(3.6714e-06)
max difference: tensor(2.5749e-05)
mean difference: tensor(3.7524e-06)
max difference: tensor(2.6703e-05)
mean difference: tensor(3.8331e-06)
max difference: tensor(2.7657e-05)
mean difference: tensor(3.9104e-06)
max difference: tensor(2.7657e-05)
mean difference: tensor(3.9891e-06)
max difference: tensor(2.7657e-05)
mean difference: tensor(4.0650e-06)
max difference: tensor(2.7180e-05)
mean difference: tensor(4.1417e-06)
max difference: tensor(2.9802e-05)
mean difference: tensor(4.2160e-06)
max difference: tensor(2.8610e-05)
mean difference: tensor(4.2916e-06)
max difference: tensor(2.9087e-05)
mean difference: tensor(4.3653e-06)
max difference: tensor(3.0160e-05)
mean difference: tensor(4.4378e-06)
max difference: tensor(3.0518e-05)
mean difference: tensor(4.5075e-06)
max difference: tensor(3.0518e-05)
mean difference: tensor(4.5811e-06)
max difference: tensor(3.0518e-05)
mean difference: tensor(4.6516e-06)
max difference: tensor(3.1471e-05)
mean difference: tensor(4.7195e-06)
max difference: tensor(3.2425e-05)
mean difference: tensor(4.7900e-06)
max difference: tensor(3.3379e-05)
mean difference: tensor(4.8578e-06)
max difference: tensor(3.4332e-05)
mean difference: tensor(4.9237e-06)
max difference: tensor(3.6240e-05)
mean difference: tensor(4.9870e-06)
max difference: tensor(3.6240e-05)
mean difference: tensor(5.0516e-06)
max difference: tensor(3.4332e-05)
mean difference: tensor(5.1189e-06)
max difference: tensor(3.6240e-05)
mean difference: tensor(5.1836e-06)
max difference: tensor(3.6240e-05)
mean difference: tensor(5.2470e-06)
max difference: tensor(4.0054e-05)
mean difference: tensor(7.6511e-07)
max difference: tensor(5.7220e-06)
```


## CTRL权重，带有mask。
```python
compare loss:
mean difference: tensor(9.5367e-07)
max difference: tensor(9.5367e-07)
compare logits:
mean difference: tensor(4.7009e-06)
max difference: tensor(0.0003)
compare hidden_states:
mean difference: tensor(1.5766e-07)
max difference: tensor(7.6294e-06)
mean difference: tensor(1.9916e-06)
max difference: tensor(0.0002)
mean difference: tensor(2.8432e-06)
max difference: tensor(0.0002)
mean difference: tensor(3.3660e-06)
max difference: tensor(0.0002)
mean difference: tensor(4.3964e-06)
max difference: tensor(0.0003)
mean difference: tensor(5.8498e-06)
max difference: tensor(0.0007)
mean difference: tensor(8.1426e-06)
max difference: tensor(0.0012)
mean difference: tensor(1.0173e-05)
max difference: tensor(0.0014)
mean difference: tensor(1.2193e-05)
max difference: tensor(0.0015)
mean difference: tensor(1.4426e-05)
max difference: tensor(0.0015)
mean difference: tensor(1.7794e-05)
max difference: tensor(0.0016)
mean difference: tensor(2.0619e-05)
max difference: tensor(0.0016)
mean difference: tensor(2.3376e-05)
max difference: tensor(0.0017)
mean difference: tensor(2.6997e-05)
max difference: tensor(0.0017)
mean difference: tensor(3.0182e-05)
max difference: tensor(0.0017)
mean difference: tensor(3.4585e-05)
max difference: tensor(0.0017)
mean difference: tensor(3.9950e-05)
max difference: tensor(0.0018)
mean difference: tensor(4.5901e-05)
max difference: tensor(0.0022)
mean difference: tensor(5.2203e-05)
max difference: tensor(0.0025)
mean difference: tensor(5.8097e-05)
max difference: tensor(0.0029)
mean difference: tensor(6.3978e-05)
max difference: tensor(0.0034)
mean difference: tensor(6.8639e-05)
max difference: tensor(0.0036)
mean difference: tensor(7.3154e-05)
max difference: tensor(0.0044)
mean difference: tensor(7.8685e-05)
max difference: tensor(0.0046)
mean difference: tensor(8.2973e-05)
max difference: tensor(0.0038)
mean difference: tensor(8.8217e-05)
max difference: tensor(0.0042)
mean difference: tensor(9.2516e-05)
max difference: tensor(0.0035)
mean difference: tensor(9.6756e-05)
max difference: tensor(0.0040)
mean difference: tensor(0.0001)
max difference: tensor(0.0040)
mean difference: tensor(0.0001)
max difference: tensor(0.0042)
mean difference: tensor(0.0001)
max difference: tensor(0.0046)
mean difference: tensor(0.0001)
max difference: tensor(0.0048)
mean difference: tensor(0.0001)
max difference: tensor(0.0051)
mean difference: tensor(0.0001)
max difference: tensor(0.0052)
mean difference: tensor(0.0001)
max difference: tensor(0.0056)
mean difference: tensor(0.0001)
max difference: tensor(0.0057)
mean difference: tensor(0.0001)
max difference: tensor(0.0062)
mean difference: tensor(0.0001)
max difference: tensor(0.0063)
mean difference: tensor(0.0001)
max difference: tensor(0.0067)
mean difference: tensor(0.0001)
max difference: tensor(0.0067)
mean difference: tensor(0.0001)
max difference: tensor(0.0069)
mean difference: tensor(0.0002)
max difference: tensor(0.0071)
mean difference: tensor(0.0002)
max difference: tensor(0.0074)
mean difference: tensor(0.0002)
max difference: tensor(0.0079)
mean difference: tensor(0.0002)
max difference: tensor(0.0087)
mean difference: tensor(0.0002)
max difference: tensor(0.0092)
mean difference: tensor(0.0002)
max difference: tensor(0.0100)
mean difference: tensor(0.0002)
max difference: tensor(0.0134)
mean difference: tensor(1.2011e-06)
max difference: tensor(5.8532e-05)
```