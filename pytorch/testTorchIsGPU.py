import torch
st = torch.cuda.is_available()  ## 输出应该是True
print("torch可用状态为："+str(st))
t = torch.nn.Linear(3,3)
t.to("cuda:0")

input = torch.randn((3,3)).requires_grad_().to("cuda:0")
output = t(input)

loss = torch.sum(output)
torch.autograd.grad(loss,input,retain_graph=True)  ## 输出应该是一个gpu上的梯度矩阵
loss.backward()
