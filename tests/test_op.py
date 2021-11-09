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

    if get_diff(y_cpu,y_dev) > 1e-3:
        raise Exception("Diff too big")

def test_fwd_bwd(inputs,call,device):
    xs_cpu = []
    xs_dev = []
    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = torch.randn(s)
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
    

def test_all(device):
    print("Mean 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=0,keepdim=True),device)
    print("Mean 2d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=(1,2),keepdim=True),device)
    print("Mean 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=0,keepdim=False),device)
    print("Mean 2d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=(0,2),keepdim=False),device)

    print("Sum 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=0,keepdim=True),device)
    print("Sum 2d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=(1,2),keepdim=True),device)
    print("Sum 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=0,keepdim=False),device)
    print("Sum 2d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=(0,2),keepdim=False),device)

    print("LogSoftmax")
    test_fwd_bwd([([4,3],-1)],torch.nn.LogSoftmax(dim=1),device)
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
    print("Tanh")
    test_fwd_bwd([([4,3],-1)],torch.nn.Tanh(),device)
    print("Tanh_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.tanh_(x*1.0),device)
    print("SiLU")
    test_fwd_bwd([([4,3],-1)],torch.nn.SiLU(),device)
    print("SiLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.SiLU(inplace=True)(x*1.0),device)



if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--device',default='opencl:0')
    r = p.parse_args()
    if r.device.find('opencl')==0:
        torch.ops.load_library("build/libpt_ocl.so")
    test_all(r.device)
