import torch
import numpy as np
torch.ops.load_library("build/libpt_ocl.so")
if False:
    x=torch.randn(10,requires_grad=True)
    y=torch.ops.my_ops.artik_op(x)
    print("FWD!!!!!!!!! Done",y)
    dy=torch.randn(10)
    y.backward(dy,retain_graph=True)
    print('x=',x)
    print('y=',y)
    print('dy=',dy)
    print('dx=',x.grad)

dev='privateuseone:0'

if False:
    t1=torch.ones((20,10),requires_grad=True,device=dev)
    t2=torch.randn(1,10).to(dev)
    with torch.no_grad():
        print(t1.shape)
        print(t2.shape)
        tc = t1.to('cpu')
        print(tc.shape)
        print(tc)
        tc = t2.to('cpu')
        print(tc)
        t3 = t1 + t2
        print(t3.to('cpu'))

grid_src = torch.randn(2,3,4);
grid_dev = grid_src.detach().clone().to(dev)
ref = grid_src.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
dev = grid_dev.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
print("REF")
print(ref)
print("DEV")
print(dev)
print(np.max(np.abs(ref.astype(np.float32)-dev.astype(np.float32))))
