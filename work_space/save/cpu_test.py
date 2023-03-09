#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
# @Time: 2021-09-14 20:44
# @File:  cpu_test.py
# @Author:  Carl
# ================================================================
import torch

path = r"./model_final.pth"
# model = torch.load(path, map_location=lambda storage, loc: storage.cuda)
# cpu="False"if torch.cuda.is_available() else "True"
cpu="False"

print(cpu)
cuda = lambda storage, loc: storage.cuda
# print(cuda)
# print(type(cuda))
if cpu == "False":
    model = torch.load(path,map_location='cpu')
# else:
#     model = torch.load(path, map_location=lambda storage, loc: storage.cuda)
# print(model)

# k 参数名 v 对应参数值
# for k, v in model.items():
#     print("k" * 10)
#     print(k)
#     print("v" * 10)
#     print(v)
