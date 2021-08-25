from transformers import CTRLConfig, CTRLLMHeadModel
from collections import OrderedDict

dont_transpose = [
    ".w.weight",
    "layernorm1.weight",
    "layernorm2.weight",
    "layernorm.weight",
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path):
    import paddle
    import torch

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        if k == "lm_head.weight":
            continue

        transpose = False
        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    transpose = True
        if k == "lm_head.bias":
            k = "lm_head_bias"
        print(f"Converting: {k} | is_transpose {transpose}")

        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


config = CTRLConfig.from_pretrained("original_ctrl/config.json")
model = CTRLLMHeadModel(config)
model.save_pretrained("random_ctrl")

# 转换random的权重。
convert_pytorch_checkpoint_to_paddle(
    pytorch_checkpoint_path="random_ctrl/pytorch_model.bin",
    paddle_dump_path="random_ctrl/model_state.pdparams",
)

