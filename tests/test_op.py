import torch
import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
import sys
import torch.nn.functional as F
import torch.nn as nn


def get_diff(cpu,dev):
    c_dev = dev.to('cpu')
    r = torch.max(torch.abs(cpu - c_dev)).item()
    if r > 1e-4:
        print(cpu)
        print(c_dev)
    return r

def test_fwd(inputs,call,device):
    xs_cpu = []
    xs_dev = []
    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = torch.randn(s)
                x_dev = x_cpu.to(device)
            else:
                x_cpu = torch.randint(limit,s)
                x_dev = x_cpu.to(device)
            xs_cpu.append(x_cpu)
            xs_dev.append(x_dev)

    y_cpu = call(*xs_cpu)
    y_dev = call(*xs_dev)

    if get_diff(y_cpu,y_dev) > 1e-6:
        raise Exception("Diff too big")
    print("Ok")

def test_fwd_bwd_op(inputs,call,device,randgen=torch.randn,paramgen = None):
    xs_cpu = []
    xs_dev = []
    p_names = set()
    with torch.no_grad():
        p_names = set(call.state_dict())
        if paramgen is not None:
            for p in call.parameters():
                size = list(p.shape)
                tmp = paramgen(size,dtype=p.dtype)
                p.copy_(tmp)

    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = randgen(s)
                x_dev = x_cpu.to(device)
                x_cpu.requires_grad = True
                x_dev.requires_grad = True
            else:
                x_cpu = torch.randint(limit,s)
                x_dev = x_cpu.to(device)
            xs_cpu.append(x_cpu)
            xs_dev.append(x_dev)


    y_cpu = call(*xs_cpu)

    with torch.no_grad():
        dy_cpu = torch.randn(y_cpu.shape)
        dy_dev = dy_cpu.to(device)
    y_cpu.backward(dy_cpu,retain_graph=True)

    dW_cpu = {}
    for n in p_names:
        dW_cpu[n] = call.get_parameter(n).grad.detach().clone()

    call.zero_grad()
    call = call.to(device)
    y_dev = call(*xs_dev)
    y_dev.backward(dy_dev,retain_graph=True)
    dW_dev = {}
    for n in p_names:
        dW_dev[n] = call.get_parameter(n).grad.detach().clone()


    with torch.no_grad():
        diffs = []
        diffs.append(('y',get_diff(y_cpu,y_dev)))
        for i in range(len(inputs)):
            if inputs[i][1] <= 0:
                diffs.append(('x%d' % i ,get_diff(xs_cpu[i].grad,xs_dev[i].grad)))
        for name in p_names:
            diffs.append(('p_' + name,get_diff(dW_cpu[name],dW_dev[name])))

        diffs.sort(key=lambda x:x[1],reverse=True)

    for name,diff in diffs:
        print("%10s %f" % (name,diff))
    max_diff = diffs[0][1]
    if max_diff > 1e-3:
        raise Exception("Diff too big")


def test_fwd_bwd(inputs,call,device,randgen=torch.randn):
    xs_cpu = []
    xs_dev = []
    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = randgen(s)
                x_dev = x_cpu.to(device)
                x_cpu.requires_grad = True
                x_dev.requires_grad = True
            else:
                x_cpu = torch.randint(limit,s)
                x_dev = x_cpu.to(device)
            xs_cpu.append(x_cpu)
            xs_dev.append(x_dev)

    y_cpu = call(*xs_cpu)
    y_dev = call(*xs_dev)

    if y_cpu.shape:
        with torch.no_grad():
            dy_cpu = torch.randn(y_cpu.shape)
            dy_dev = dy_cpu.to(device)
        y_cpu.backward(dy_cpu,retain_graph=True)
        y_dev.backward(dy_dev,retain_graph=True)
    else:
        y_cpu.backward(retain_graph=True)
        y_dev.backward(retain_graph=True)


    with torch.no_grad():
        diffs = []
        diffs.append(('y',get_diff(y_cpu,y_dev)))
        for i in range(len(inputs)):
            if inputs[i][1] <= 0:
                diffs.append(('x%d' % i ,get_diff(xs_cpu[i].grad,xs_dev[i].grad)))

        diffs.sort(key=lambda x:x[1],reverse=True)

    for name,diff in diffs:
        print("%10s %f" % (name,diff))
    max_diff = diffs[0][1]
    if max_diff > 1e-3:
        raise Exception("Diff too big")
    


def get_mat(D0,D1,d,ind):
    tr = ind % 2
    offset = 0 if ind < 2 else 5
    non_cont = ind >= 4
    if tr:
        D0,D1=D1,D0
    tmp = torch.arange(0,400,device=d,dtype=torch.float32)
    size = D0*D1
    if non_cont:
        size *= 4
    tmp = tmp[offset:offset+size]
    if non_cont:
        tmp = tmp.view(D0*2,D1*2)
        tmp = tmp[::2,::2]
    else:
        tmp = tmp.view(D0,D1)
    if tr:
        tmp = tmp.T
    return tmp

def  test_mm(device):
    print("Tesing mm")
    M = 4
    N = 10
    K = 6
    re = 110
    for Ct in range(6):
        for At in range(6):
            for Bt in range(6):
                for d in ["cpu",device]:
                    A = get_mat(M,K,d,At)
                    B = get_mat(K,N,d,Bt)
                    C = get_mat(M,N,d,Ct)
                    torch.mm(A,B,out=C)
                    if d == "cpu":
                        C_ref = C
                if get_diff(C_ref,C) != 0:
                    print("Failed for",M,N,K,At,Bt,Ct);
                    raise Exception("Failure")
    print("Ok")

def get_bmat(B,D0,D1,d,ind):
    tr = ind % 2
    offset = 0 if ind < 2 else 5
    non_cont = ind >= 4
    if tr:
        D0,D1=D1,D0
    tmp = torch.arange(0,10+B*D0*D1*4,device=d,dtype=torch.float32)
    size = B*D0*D1
    if non_cont:
        size *= 4
    tmp = tmp[offset:offset+size]
    if non_cont:
        tmp = tmp.view(B,D0*2,D1*2)
        tmp = tmp[:,::2,::2]
    else:
        tmp = tmp.view(B,D0,D1)
    if tr:
        tmp = torch.transpose(tmp,2,1)
    return tmp

def test_bmm(device):
    print("Tesing bmm")
    Bc = 3
    M = 4
    N = 10
    K = 6
    for Ba,Bb in [(Bc,Bc),(1,Bc),(Bc,1)]:
        for Ct in range(2):
            for At in range(6):
                for Bt in range(6):
                    for d in ["cpu",device]:
                        A = get_bmat(Ba,M,K,d,At)
                        B = get_bmat(Bb,K,N,d,Bt)
                        C = get_bmat(Bc,M,N,d,Ct)
                        if Ba < Bc:
                            A=A.expand(Bc,M,K)
                        if Bb < Bc:
                            B=B.expand(Bc,K,N)
                        torch.bmm(A,B,out=C)
                        if d == "cpu":
                            C_ref = C
                    if get_diff(C_ref,C) != 0:
                        print("Failed for",M,N,K,At,Bt,Ct);
                        raise Exception("Failure")
    print("Ok")

    

def test_all(device):
    print("Mean 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=0,keepdim=True),device)
    print("Mean 2d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=(1,2),keepdim=True),device)
    print("Mean 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=0,keepdim=False),device)
    print("Mean 2d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=(0,2),keepdim=False),device)
    print("Mean all squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x),device)

    print("Sum 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=0,keepdim=True),device)
    print("Sum 2d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=(1,2),keepdim=True),device)
    print("Sum 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=0,keepdim=False),device)
    print("Sum 2d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=(0,2),keepdim=False),device)

    print("Prod 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.prod(x,dim=1,keepdim=True),device)
    print("Prod 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.prod(x,dim=1,keepdim=False),device)


    print("LogSoftmax 2d")
    test_fwd_bwd([([4,3],-1)],torch.nn.LogSoftmax(dim=1),device)
    print("LogSoftmax 3d last")
    test_fwd_bwd([([4,3,5],-1)],torch.nn.LogSoftmax(dim=-1),device)
    print("LogSoftmax 3d midle")
    test_fwd_bwd([([4,3,5],-1)],torch.nn.LogSoftmax(dim=1),device)
    print("LogSoftmax 3d 1st")
    test_fwd_bwd([([4,3,5],-1)],torch.nn.LogSoftmax(dim=0),device)
    print("Softmax")
    test_fwd_bwd([([4,3],-1)],torch.nn.Softmax(dim=1),device)
    print("NLLLoss");
    test_fwd_bwd([([4,3],-1),([4],3)],torch.nn.NLLLoss(),device)
    print("AAPool2d")
    test_fwd_bwd([([4,8,2,2],-1)],torch.nn.AdaptiveAvgPool2d((1,1)),device)
    print("Abs")
    test_fwd_bwd([([4,3],-1)],torch.abs,device)
    print("Abs_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.abs_(x*1.0),device)
    print("Hardtanh")
    test_fwd_bwd([([4,3],-1)],torch.nn.Hardtanh(),device)
    print("Hardtanh_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.Hardtanh(inplace=True)(x*1.0),device)
    print("Sigmoid")
    test_fwd_bwd([([4,3],-1)],torch.nn.Sigmoid(),device)
    print("Sigmoid_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.sigmoid_(x*1.0),device)
    print("Hardsigmoid")
    test_fwd_bwd([([4,3],-1)],torch.nn.Hardsigmoid(),device)
    print("Hardsigmoid_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.Hardsigmoid(inplace=True)(x*1.0),device)
    print("ReLU")
    test_fwd_bwd([([4,3],-1)],torch.nn.ReLU(),device)
    print("ReLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.ReLU(inplace=True)(x*1.0),device)
    print("LReLu")
    test_fwd_bwd([([4,3],-1)],torch.nn.LeakyReLU(),device)
    print("LReLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.LeakyReLU(inplace=True)(x*1.0),device)
    print("Tanh")
    test_fwd_bwd([([4,3],-1)],torch.nn.Tanh(),device)
    print("Tanh_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.tanh_(x*1.0),device)
    print("SiLU")
    test_fwd_bwd([([4,3],-1)],torch.nn.SiLU(),device)
    print("SiLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.SiLU(inplace=True)(x*1.0),device)

    print("GELU")
    test_fwd_bwd([([4,3],-1)],torch.nn.GELU(),device)
    print("GELU tanh")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.GELU(approximate='tanh')(x*1.0),device)

    print("atan")
    test_fwd([([4,3,5],-1)],torch.atan,device)

    #print("ChannelShuffle")
    #test_fwd_bwd([([3, 4, 2, 2],-1)],torch.nn.ChannelShuffle(2),device)
    print("BCE Loss")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.BCELoss(),device,torch.rand)
    print("BCE Loss no reduction")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.BCELoss(reduction='none'),device,torch.rand)
    print("MSE Loss")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.MSELoss(),device,torch.rand)
    print("MSE Loss no reduction")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.MSELoss(reduction='none'),device,torch.rand)
    print("Min")
    test_fwd([([4,3,5],-1)],torch.min,device)
    print("Max")
    test_fwd([([4,3,5],-1)],torch.max,device)
    print("Lerp")
    test_fwd([([4,3,5],-1),([4,3,1],-1)],lambda x,y:torch.lerp(x,y,0.1),device)
    print("Dot")
    test_fwd([([16],-1),([16],-1)],torch.dot,device)
    print("Clamp 1")
    test_fwd([([4,3,5],-1)],lambda x:torch.clamp(x,min=-0.2,max=0.3),device)
    print("Clamp 2")
    test_fwd([([4,3,5],-1)],lambda x:torch.clamp(x,min=-0.2),device)
    print("Clamp 3")
    test_fwd([([4,3,5],-1)],lambda x:torch.clamp(x,max=0.3),device)

    print("Linear 2d")
    test_fwd_bwd_op([([8,10],-1)],torch.nn.Linear(10,5),device)
    print("Linear 3d")
    test_fwd_bwd_op([([2,6,10],-1)],torch.nn.Linear(10,5),device)

    print("Conv")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.Conv2d(6,8,[3,5],stride=[1,2],padding=[1,2],dilation=1,groups=2),device)
    print("ConvTr")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.ConvTranspose2d(6,8,[3,5],stride=[1,2],padding=[1,2],dilation=1,groups=2),device)
    print("ConvTr pad")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.ConvTranspose2d(6,8,[3,5],stride=[1,2],padding=[1,2],output_padding=[0,1],dilation=1,groups=2),device)
    print("ConvTr")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.ConvTranspose2d(6,8,[3,5],stride=[1,2],padding=[1,2],dilation=1,groups=1,bias=False),device)

    print("LayerNorm")
    test_fwd_bwd_op([([2,3,4,30],-1)],torch.nn.LayerNorm((4,30),elementwise_affine=False),device)

    print("LayerNorm Aff")
    test_fwd_bwd_op([([2,3,4,30],-1)],torch.nn.LayerNorm((4,30),elementwise_affine=True),device,paramgen = torch.randn)

    print("Test logit eps")
    test_fwd([([4,3,5],-1)],lambda x:torch.logit(x,eps=0.1),device)

    print("Test logit")
    test_fwd([([4,3,5],-1)],lambda x:torch.logit(torch.clamp(x,min=0.1,max=0.9)),device)

    print("Test arange out")
    test_fwd([([10],-1)],lambda x:torch.arange(0,10,out=x),device)

    print("Test arange out empty")
    test_fwd([([1],-1)],lambda x:torch.arange(0,10,out=x),device)

    print("Test amax full")
    test_fwd_bwd([([2,3,4,5],-1)],lambda x:torch.amax(x,(1,-1)),device)

    print("Test amax keepdim")
    test_fwd_bwd([([2,3,4,5],-1)],lambda x:torch.amax(x,(1,-1),keepdim=True),device)

    print("Test amax all")
    test_fwd_bwd([([2,3,4,5],-1)],lambda x:torch.amax(x,keepdim=False),device)

    print("Test amin full")
    test_fwd_bwd([([2,3,4,5],-1)],lambda x:torch.amin(x,(1,-1)),device)

    print("Test amin keepdim")
    test_fwd_bwd([([2,3,4,5],-1)],lambda x:torch.amin(x,(1,-1),keepdim=True),device)

    print("Test amin all")
    test_fwd_bwd([([2,3,4,5],-1)],lambda x:torch.amin(x,keepdim=False),device)

    print("Test addmm 0/1")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1)],lambda a,b,c:torch.addmm(a,b,c,beta=0.0,alpha=1.0),device)
    print("Test addmm 1/1")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1)],lambda a,b,c:torch.addmm(a,b,c,beta=1.0,alpha=1.0),device)
    print("Test addmm 1.0/0.5")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1)],lambda a,b,c:torch.addmm(a,b,c,beta=1.0,alpha=0.5),device)
    print("Test addmm 0.5/0.5")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1)],lambda a,b,c:torch.addmm(a,b,c,beta=0.5,alpha=0.5),device)

    print("Test addmm 0/1 cp")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1),([4,2],-1)],lambda a,b,c,o:torch.addmm(a,b,c,beta=0.0,alpha=1.0,out=o.T),device)
    print("Test addmm 1/1 cp")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1),([4,2],-1)],lambda a,b,c,o:torch.addmm(a,b,c,beta=1.0,alpha=1.0,out=o.T),device)
    print("Test addmm 1.0/0.5 cp")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1),([4,2],-1)],lambda a,b,c,o:torch.addmm(a,b,c,beta=1.0,alpha=0.5,out=o.T),device)
    print("Test addmm 0.5/0.5 cp")
    test_fwd([([2,4],-1),([2,3],-1),([3,4],-1),([4,2],-1)],lambda a,b,c,o:torch.addmm(a,b,c,beta=0.5,alpha=0.5,out=o.T),device)

    print("Test _transform_bias_rescale_qkv")
    test_fwd([([5,7,3*2*11],-1),([3*2*11],-1)],lambda a,b:torch.cat(torch._transform_bias_rescale_qkv(a,b,2),dim=0),device)

    print("Test interpolate nearest factors")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,scale_factor=(2.3,1.5),mode='nearest'),device)

    print("Test interpolate nearest size")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,size=(20,20),mode='nearest'),device)

    print("Test interpolate nearest factors")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,scale_factor=(2.3,1.5),mode='nearest-exact'),device)

    print("Test interpolate nearest size")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,size=(20,20),mode='nearest-exact'),device)

    print("Test interpolate bilinear factors")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,scale_factor=(2.3,1.5),mode='bilinear'),device)

    print("Test interpolate bilinear size")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,size=(20,20),mode='bilinear'),device)

    print("Test interpolate bilinear factors align_corners")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,scale_factor=(2.3,1.5),mode='bilinear',align_corners = True),device)

    print("Test interpolate bilinear size align corners")
    test_fwd_bwd([([2,3,10,15],-1)],lambda x:torch.nn.functional.interpolate(x,size=(20,20),mode='bilinear',align_corners=True),device)

def test_concat(dev):
    print("Test concat")
    with torch.no_grad():
        x1 = torch.randn(5,10,device='cpu')
        x2 = torch.randn(5,20,device='cpu')
        y1 = torch.concat((x1,x2),dim=1)
        y2 = torch.zeros(5,30,device='cpu')
        torch.concat((x1,x2),dim=1,out=y2)
        y3s = torch.zeros(10,30,device='cpu')
        y3 = y3s[::2,:]
        torch.concat((x1,x2),dim=1,out=y3)

        x1d = x1.to(dev)
        x2d = x2.to(dev)
        y1d = torch.concat((x1d,x2d),dim=1)
        y2d = torch.zeros(5,30,device=dev)
        torch.concat((x1d,x2d),dim=1,out=y2d)
        y3sd = y3s.to(dev)
        y3d = y3sd[::2,:]
        torch.concat((x1d,x2d),dim=1,out=y3d)
   
        for n,d in [('direct',get_diff(y1,y1d)),
                    ('out',get_diff(y2,y2d)),
                    ('out/s base',get_diff(y3s,y3sd)),
                    ('outs/',get_diff(y3,y3d))]:
            print("%10s %.5f" % (n,d))
            if d > 0:
                raise Exception("Failed concat:" + n)

def seed_dev(dev,s):
    if dev.find('privateuseone') == 0:
        import pytorch_ocl
        pytorch_ocl.manual_seed_all(s)
    else:
        torch.manual_seed(s)

def test_rng(dev):
    print("Testing RNG")
    seed_dev(dev,10)
    x1=torch.randn(10,device=dev)
    x2=torch.randn(10,device=dev)
    seed_dev(dev,10)
    x3=torch.randn(10,device=dev)
    x1 = x1.cpu()
    x2 = x2.cpu()
    x3 = x3.cpu()
    assert torch.max(torch.abs(x1-x2)).item() > 0
    assert torch.max(torch.abs(x1-x3)).item() == 0
    print("RNG Ok")
    print("Testing dropout")
    x=torch.randn(1000,device=dev)
    x.requires_grad=True
    y=torch.nn.functional.dropout(x,p=0.25,training=False);
    assert torch.max(torch.abs(x-y)).item() ==0
    x=torch.rand(1000,device=dev)
    x.requires_grad=True
    y=torch.nn.functional.dropout(x,p=0.25,training=True);
    assert abs(torch.mean((y > 0).to(torch.float32)).item() - 0.75) < 0.1
    assert (torch.mean(x) - torch.mean(y)).item() < 0.05
    y.backward(torch.ones(1000,device=dev))
    dx = x.grad
    assert abs(torch.mean((dx > 0).to(torch.float32)).item() - 0.75) < 0.1
    print("Dropout ok")


def test_logical(device):
    l_bool_int = [
            ('&',lambda a,b:(a&b)),
            ('|',lambda a,b:(a|b)),
            ('^',lambda a,b:(a^b)),
            ('~',lambda a,b:(~a))]
    for with_bool,cases in [(True,l_bool_int)]:
        for name,op in cases:
            print("Test ",name)
            test_fwd([([20,30],10),([1,30],10)],lambda a,b:op(a-5,a-5),device)
            if with_bool:
                print("Test ",name, " with bool")
                test_fwd([([20,30],10),([1,30],10)],lambda a,b:op(a>5,a>5).to(torch.float32),device)


def test_comp(device):
    l1 = [("maximum",torch.maximum),("minimum",torch.minimum)]
    l2 = [('==',torch.eq),('!=',torch.ne),('>',torch.gt),('<',torch.lt),('>=',torch.ge),('<=',torch.le)]
    for with_scal,cases in [(True,l2),(False,l1)]:
        for name,op in cases:
            print("Test ",name)
            test_fwd([([20,30],-1),([1,30],-1)],lambda a,b:op(torch.round(a)*10,torch.round(b)*10).to(torch.float),device)
            if with_scal:
                print("Test ",name, " with scalar")
                test_fwd([([20,30],-1)],lambda a:op(torch.round(a)*10,5).to(torch.float),device)
            print("Test ",name, " with tenspor R")
            test_fwd([([20,30],-1)],lambda a:op(torch.round(a)*10,torch.tensor(5,dtype=torch.float32)).to(torch.float),device)
            print("Test ",name, " with tenspor L")
            test_fwd([([20,30],-1)],lambda a:op(torch.tensor(5,dtype=torch.float32),torch.round(a)*10).to(torch.float),device)
    for name,op in [('>',lambda x:x*3>0.5),
               ('<',lambda x:x*3<0.5),
               ('==',lambda x:torch.round(x*3)==1),
               ('!=',lambda x:torch.round(x*3)==1),
               ('<=',lambda x:torch.round(x*3)>=1),
               ('>=',lambda x:torch.round(x*3)<=1)]:
        print(f" Testng x{name}val")
        test_fwd([([4,4],-1)],lambda x:op(x).to(torch.float32),device=device)


if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--device',default='ocl:0')
    r = p.parse_args()
    if r.device.find('ocl')==0 or r.device.find('privateuseone')==0:
        import pytorch_ocl
    test_logical(r.device)
    test_comp(r.device)
    test_all(r.device)
    test_concat(r.device)
    test_rng(r.device)
    test_mm(r.device)
    test_bmm(r.device)
