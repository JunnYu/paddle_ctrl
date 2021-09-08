import paddle
import torch
import numpy as np
from hgctrl.modeling_ctrl import CTRLLMHeadModel as PTCTRLLMHeadModel
from ctrl.modeling import CTRLLMHeadModel as PDCTRLLMHeadModel

torch.set_grad_enabled(False)
paddle.set_grad_enabled(False)
paddle.set_device("gpu")
paddle.set_default_dtype("float64")

def compare(a, b):
    a = torch.tensor(a.cpu().numpy())
    b = torch.tensor(b.cpu().numpy())
    meandif = (a - b).abs().mean()
    maxdif = (a - b).abs().max()
    print("mean difference:", meandif)
    print("max difference:", maxdif)


pd = True


if pd:
    pd_model = PDCTRLLMHeadModel.from_pretrained("../paddle_ctrl/pd/ctrl") 
    pd_model.eval()
    np.random.seed(42)
    inputs = np.random.randint(0, 246534, size=(4, 128), dtype="int64")
    attention_mask = np.random.randint(1, 2, size=(4, 128), dtype="int64")

    pd_outputs = pd_model(
        input_ids=paddle.to_tensor(inputs),
        attention_mask=paddle.to_tensor(attention_mask),
        labels=paddle.to_tensor(inputs),
        output_hidden_states=True,
    )
    paddle.save(pd_outputs.logits,"logits.pd")
    paddle.save(pd_outputs.hidden_states,"hidden_states.pd")
else:
    np.random.seed(42)
    inputs = np.random.randint(0, 246534, size=(4, 128), dtype="int64")
    attention_mask = np.random.randint(1, 2, size=(4, 128), dtype="int64")
    pt_model = PTCTRLLMHeadModel.from_pretrained("../paddle_ctrl/hg/ctrl").double() 
    pt_model.eval()
    pt_model.cuda()
    pt_outputs = pt_model(
        input_ids=torch.tensor(inputs).cuda(),
        attention_mask=torch.tensor(attention_mask).cuda(),
        labels=torch.tensor(inputs).cuda(),
        output_hidden_states=True,
    )
    torch.save(pt_outputs.logits,"logits.pt")
    torch.save(pt_outputs.hidden_states,"hidden_states.pt")
