import paddle
import torch
import numpy as np
from transformers import CTRLLMHeadModel as PTCTRLLMHeadModel
from ctrl.modeling import CTRLLMHeadModel as PDCTRLLMHeadModel

torch.set_grad_enabled(False)
paddle.set_grad_enabled(False)
paddle.set_device("cpu")


def compare(a, b):
    a = torch.tensor(a.numpy()).float()
    b = torch.tensor(b.numpy()).float()
    meandif = (a - b).abs().mean()
    maxdif = (a - b).abs().max()
    print("mean difference:", meandif)
    print("max difference:", maxdif)


pd_model = PDCTRLLMHeadModel.from_pretrained("pd/sshleifer-tiny-ctrl")
pd_model.eval()
pt_model = PTCTRLLMHeadModel.from_pretrained("hg/sshleifer-tiny-ctrl")
pt_model.eval()

inputs = np.random.randint(0, 246534, size=(4, 128), dtype="int64")
attention_mask = np.random.randint(1, 2, size=(4, 128), dtype="int64")

pd_outputs = pd_model(
    input_ids=paddle.to_tensor(inputs),
    attention_mask=paddle.to_tensor(attention_mask),
    labels=paddle.to_tensor(inputs),
    output_hidden_states=True,
)

pt_outputs = pt_model(
    input_ids=torch.tensor(inputs),
    attention_mask=torch.tensor(attention_mask),
    labels=torch.tensor(inputs),
    output_hidden_states=True,
)


print("compare loss:")
compare(pd_outputs.loss, pt_outputs.loss)
print("compare logits:")
compare(pd_outputs.logits, pt_outputs.logits)
print("compare hidden_states:")
for a, b in zip(pd_outputs.hidden_states, pt_outputs.hidden_states):
    compare(a, b)

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