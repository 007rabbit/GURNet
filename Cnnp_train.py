import torch
import torch.nn as nn
import torch.optim as optim
import Cnnp_Net
import numpy as np
from torchvision import datasets, transforms
import argparse
import torch.utils.data
import torch.nn.functional as F  # 小写变量导入为非小写
import time
import random
# from log import *
# logger = configure_logging()
'''0,255像素范围内训练'''

parser = argparse.ArgumentParser(description='train Cnnp')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for trainning')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate of train')
parser.add_argument('--weight_decay', type=float, default=0.001, metavar='wd',
                    help='weight_deacy')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='Adam BETA paramters.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 判断GPU是否可用-----is_available
torch.manual_seed(args.seed)  # 设置cpu随机种子,生成随机数
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 设置gpu随机种子
else:
    args.gpu = None
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}  # pin_memory拷贝数据到GPU,num_workers设置工作进程
def custom_rotation(img):
    # 生成90度的倍数的旋转角度
    angle = random.randint(0, 3) * 90
    return transforms.functional.rotate(img, angle)
'''训练过程'''
train_path2 = r"D:\haiyang\datasets\BOWS_256_256_3000"  # 设置路径
train_data = datasets.ImageFolder(train_path2, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(custom_rotation),
    transforms.ToTensor(),  # transforms.ToTensor()将像素值从整数范围[0, 255]转换为浮点数范围[0, 1]
]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
'''验证集'''
valid_path2 = r"D:\haiyang\datasets\valid_600"   # 设置路径
valid_data = datasets.ImageFolder(valid_path2, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
]))
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=8, shuffle=False, **kwargs)
# 转化变量为元组，**为字典
model = Cnnp_dnet.ME_GGC_D_Unet() # 实例化网络参数
if args.cuda:
    model.cuda()
optimer = optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, eps=1e-08,
                     weight_decay=args.weight_decay, amsgrad=False)  # 选用Adam优化器


def pre_img(img,mask, flag):
    device = img.device
    img1 = img.clone().to(device)
    '''创建索引分量'''
    I1 = torch.zeros(img.shape, device=device)
    I2 = torch.zeros(img.shape, device=device)
    I1[:, :, :, :] = img1[:, :, :, :]*(mask == flag)
    I2[:, :, :, :] = img1[:, :, :, :]*(mask != flag)
    return I1, I2


def result_deal(img,mask, flag):
    device = img.device
    img1 = img.clone().to(device)
    I1 = torch.zeros(img.shape, device=device)
    I1[:, :, :, :] = img1[:, :, :, :] * (mask == flag)  # original 类别1
    return I1


def train(epoch):
    lr_train = (optimer.state_dict()['param_groups'][0]['lr'])
    print('lr_train=', lr_train)
    model.train()  # 激活训练模型
    '''使用卷积核进行预处理'''
    torch.autograd.set_detect_anomaly(True)
    for idx, (data, lable) in enumerate(train_loader):
        flag = random.randint(1, 4)
        x_, y_ = np.random.randint(0,128 , size=2)
        data = ((data*255)[:, :, x_:x_ + 128, y_:y_ + 128]).cuda()
        mask = category[x_:x_ + 128, y_:y_ + 128].cuda()
        i1, img = pre_img(data,mask, flag)
        img.requires_grad = True  # 张量自动求梯度，替换掉Variable()用法
        data2 = model(img)
        pre_ = result_deal(data2,mask, flag)  # 返回I1的预测值
        loss = F.smooth_l1_loss(pre_, i1, beta=5 )
        optimer.zero_grad()  # 全局梯度为0
        loss.backward()
        optimer.step()
        '''反向传播'''
        if (idx + 1) % 75 == 0:
            print('Train Epoch: {}   [{}/{} ({:.0f}%)], loss={}'.format(
                epoch, (idx + 1) * len(data), len(train_loader.dataset),
                       100. * (idx + 1) / len(train_loader), loss))
    return loss.item()


def valid():
    model.eval()
    loss0, loss1, loss2 = 0, 0, 0
    x0, x1, x2 = 0, 0, 0
    for idx, (data, label) in enumerate(valid_loader):
        data = (data*255).cuda()
        flag = random.randint(1, 4)
        """0点1叉"""
        with torch.no_grad():
            i1, img = pre_img(data, mask, flag)
            data2 = model(img)
            pre_ = result_deal(data2, mask, flag)
            loss = F.smooth_l1_loss(pre_, i1, beta=5)
            if label[0]==0:
                loss0 +=loss.item()
                x0+=1
            elif label[0]==1:
                loss1 +=loss.item()
                x1+=1
            elif label[0] == 2:
                loss2 += loss.item()
                x2 += 1
    print("avg-valid-loss=", (loss0+loss1+loss2)/(x0+x1+x2))
    return loss0/x0, loss1/x1, loss2/x2

category = torch.zeros((256,256), dtype=torch.long, device='cuda')
category[::2, ::2] = 1
category[::2, 1::2] = 2
category[1::2, ::2] = 3
category[1::2, 1::2] = 4
mask = torch.zeros((128,128), dtype=torch.long, device='cuda')
mask[::2, ::2] = 1
mask[::2, 1::2] = 2
mask[1::2, ::2] = 3
mask[1::2, 1::2] = 4

import datetime
import pandas as pd
if __name__ == '__main__':
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print('程序启动******', "日期和时间：", formatted_datetime)
    t3 = time.time()
    lo = []
    print('当前学习率',optimer.param_groups[0]['lr'])
    for epoch in range(1,2001):
        t1 = time.time()
        if epoch%400==0 :optimer.param_groups[0]['lr'] = optimer.param_groups[0]['lr']*0.5
        loss00 = train(epoch)
        loss0, loss1, loss2 = valid()
        lo.append({'loss-image': loss0, 'loss-bows': loss1, 'loss-ucid': loss2, 'idx': epoch,'lr':optimer.param_groups[0]['lr'],
                   'time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'train-loss':loss00,'Avg_loss':(loss0+loss1+loss2)/3})
        df = pd.DataFrame(lo)
        df.to_excel('loss.xlsx', index=False)
        '''loss保存，参数保存'''
        if epoch>=400:
            state = {'network': model.state_dict(), 'lr': optimer.param_groups[0]['lr']}
            torch.save(state, './train_param/' + str(epoch) + '_model.pth')
        t2 = time.time()
        print('total time = ', (t2-t1)//60, '分钟', (t2-t1) % 60, '秒')
    # 保存每个epoch最终的损失值
    t4 = time.time()
    print('total time = ', (t4-t3)//3600, '小时', ((t4-t3) % 3600)//60, '分钟', (t4-t3) % 60, '秒')
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print("当前日期和时间：", formatted_datetime)


