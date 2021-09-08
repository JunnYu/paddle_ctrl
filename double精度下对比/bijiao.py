import os
import torch
import paddle
def compare(a, b):
    a = torch.from_numpy(a.cpu().numpy())
    b = torch.from_numpy(b.cpu().numpy())
    meandif = (a - b).abs().mean().item()
    maxdif = (a - b).abs().max().item()
    return meandif,maxdif

paddle.set_device("cpu")
print(len(os.listdir("jilu"))//2)

with open(f"bijiaojieguo.txt","w") as f:
    for i in range(0,  len(os.listdir("jilu"))//2):  # ~25
        qaz = torch.load(f"jilu/{i}.pt")
        wsx = paddle.load(f"jilu/{i}.pd")
        mean_value,max_value = compare(qaz,wsx)
        f.write(f"==========================================\n")
        f.write(f"{i}\n")
        f.write(f"mean dif: {mean_value} \n")
        f.write(f"max dif: {max_value}")
        f.write("\n")