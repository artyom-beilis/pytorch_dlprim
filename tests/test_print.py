import torch
import sys
import numpy as np

device = sys.argv[1]
if device.find('privateuseone')==0:
    torch.ops.load_library("build/libpt_ocl.so")

with torch.no_grad():
    for k in [1,5,10,20,(2,3)]:
        print(" ========= ",k,"===========")
        print(" == Float == ")
        x=torch.randn((k),device=device)
        print(x.detach().cpu())
        print(x)

        x1=torch.nn.ReLU()(x)
        x2=torch.nn.ReLU()(-x)
        x3=x1/x2
        x4=x1/x1
        print(x1.detach().cpu(),'\n',x1)
        print(x2.detach().cpu(),'\n',x2)
        print(x3.detach().cpu(),'\n',x3)
        print(x4.detach().cpu(),'\n',x4)

        npv=np.random.randint(0,100,k)
        print(" == Int 32 == ")
        intx=torch.tensor(npv.astype(np.int32),device=device)
        print(intx.detach().cpu(),'\n',intx)

        print(" == Int 16 == ")
        intx=torch.tensor(npv.astype(np.int16),device=device)
        print(intx.detach().cpu(),'\n',intx)

        print(" == Int 64 == ")
        intx=torch.tensor(npv.astype(np.int64),device=device)
        print(intx.detach().cpu(),'\n',intx)

        print(" == uint8 == ")
        intx=torch.tensor(npv.astype(np.uint8),device=device)
        print(intx.detach().cpu(),'\n',intx)


