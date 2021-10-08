import torch
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

dev='opencl:1'

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
