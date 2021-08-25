# CTRL: A Conditional Transformer Language Model for Controllable Generation
paddle2.x 实现 [CTRL](https://arxiv.org/pdf/1909.05858.pdf)

# 准备工作
- 下载预训练权重（pytorch版本的，https://huggingface.co/ctrl/ 和 https://huggingface.co/sshleifer/tiny-ctrl）,放入hg/文件夹对应目录。
- 【可选/推荐】手动转换：`python convert.py`
- 【不推荐】百度云下载：链接：https://pan.baidu.com/s/1dnRwCRClqsXvG8475v2m9g 提取码：ogn8 

# CTRLLMHeadModel前向对齐

```bash
python compare_lm.py
############ sshleifer-tiny-ctrl
# compare loss:
# mean difference: tensor(0.)
# max difference: tensor(0.)
# compare logits:
# mean difference: tensor(1.2408e-08)
# max difference: tensor(5.6252e-07)
# compare hidden_states:
# mean difference: tensor(4.2710e-08)
# max difference: tensor(3.2783e-06)
# mean difference: tensor(4.3044e-08)
# max difference: tensor(3.2783e-06)
# mean difference: tensor(1.2185e-07)
# max difference: tensor(4.5300e-06)

############ 加载ctrl权重误差非常大,进入文件夹【探究ctrl原版权重误差大的原因】查看原因
```

# CTRLForSequenceClassification前向对齐

```bash
python compare_cls.py
############ sshleifer-tiny-ctrl
# compare loss
# mean difference: tensor(2.3842e-07)
# max difference: tensor(2.3842e-07)
# compare logits
# mean difference: tensor(6.0303e-09)
# max difference: tensor(1.4901e-08)

############ 加载ctrl权重误差非常大,进入文件夹【探究ctrl原版权重误差大的原因】查看原因
```


# tokenizer对齐
```bash
python compare_tokenizer.py
# input_ids:      True
# token_type_ids: True
# attention_mask: True
```

# Reference

```bibtex
@article{keskar2019ctrl,
  title={Ctrl: A conditional transformer language model for controllable generation},
  author={Keskar, Nitish Shirish and McCann, Bryan and Varshney, Lav R and Xiong, Caiming and Socher, Richard},
  journal={arXiv preprint arXiv:1909.05858},
  year={2019}
}
```

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```