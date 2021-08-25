import torch
from ctrl.modeling import CTRLForSequenceClassification as PDCTRLForSequenceClassification
from transformers import CTRLForSequenceClassification as PTCTRLForSequenceClassification
import paddle
import numpy as np

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
pd_model = PDCTRLForSequenceClassification.from_pretrained("pd/sshleifer-tiny-ctrl",num_labels=10)
pd_model.eval()

pt_model = PTCTRLForSequenceClassification.from_pretrained("hg/sshleifer-tiny-ctrl",num_labels=10)
pt_model.eval()

# copy classifier weights
pt_model.classifier.weight.data = torch.tensor(pd_model.classifier.weight.t().numpy())

inputs = np.random.randint(0, 246534, size=(1, 128), dtype="int64")
labels = np.array([6],dtype="int64")

pd_outputs = pd_model(paddle.to_tensor(inputs),labels=paddle.to_tensor(labels))
pt_outputs = pt_model(torch.tensor(inputs),labels=torch.tensor(labels))


print("compare loss")
compare(pd_outputs.loss,pt_outputs.loss)
print("compare logits")
compare(pd_outputs.logits,pt_outputs.logits)

# compare loss
# mean difference: tensor(2.3842e-07)
# max difference: tensor(2.3842e-07)
# compare logits
# mean difference: tensor(6.0303e-09)
# max difference: tensor(1.4901e-08)


