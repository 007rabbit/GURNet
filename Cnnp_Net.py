import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import init
import torch.nn.functional as F
import cv2
import numpy as np
'''自定义编码器'''
class GMM(nn.Module):
    def __init__(self,inc,outc):
        super(GMM, self).__init__()
        '''多感受野'''
        self.k3 = nn.Sequential(
            nn.Conv2d(inc,inc,kernel_size=3,stride=1,padding=1,groups=inc),
            nn.ReLU(inplace=True)
        )
        self.k5 = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=5, stride=1, padding=2,groups=inc),
            nn.ReLU(inplace=True)
        )
        self.k7 = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=7, stride=1, padding=3,groups=inc),
            nn.ReLU(inplace=True)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        k3 = self.k3(x)
        k5 = self.k5(x)
        k7 = self.k7(x)
        k = k3+k5+k7+x
        c1 = self.c1(k)
        c2 = self.c2(k)
        return c1*c2
'''自定义解码器'''
class MFIM(nn.Module):
    def __init__(self,inc,outc):
        super(MFIM, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(inc,inc,kernel_size=1,stride=1,padding=0),
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(inc,inc,kernel_size=1,stride=1,padding=0),
        )
        self.PW = nn.Sequential(
            nn.Conv2d(inc,outc,kernel_size=1,stride=1),
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(inc,outc,kernel_size=1,stride=1),
        )
    def forward(self,x):
        x1 = self.m1(x)
        x2 = self.m2(x)
        p1 = self.p1(x)
        pw = self.PW(x1*x2)
        return pw+p1

class GFFRM(nn.Module):
    def __init__(self,inc,outc,k):
        super(GFFRM, self).__init__()
        self.x1 = nn.Sequential(
            nn.Conv2d(inc, inc,k+2,padding=(k+2)//2),
        )
        self.x2 = nn.Sequential(
            nn.Conv2d(inc, inc,k,padding=k//2),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(inc*2, outc,k,padding=k//2,groups=inc*2),
            nn.Conv2d(inc*2, outc,1,padding=0),
        )
        self.ag = nn.Conv2d(outc, outc, k, padding=k // 2,groups=outc)
        self.param = nn.Sequential(
            nn.Conv2d(outc, outc, k, padding=k // 2,groups=outc),
            nn.Sigmoid(),
        )

    def forward(self,ed,de):
        x = self.x1(ed)
        y = self.x2(de)
        z=torch.cat([x,y],dim=1)
        fusion = self.fusion(z)
        ag = self.ag(fusion)
        p=self.param(fusion)
        return ag*p+z
'''上采样转职卷积'''
class Up_(nn.Module):
    def __init__(self,inc,outc):
        super(Up_, self).__init__()
        self.x1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=inc, out_channels=outc, kernel_size=4, stride=2, padding=1),
        )
    def forward(self,x):
        return self.x1(x)
class GURNet(nn.Module):
    def __init__(self):
        super(GURNet, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            GMM(64,64)
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            GMM(128,128)
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            GMM(128,128)
        )
        self.up3 = Up_(128, 128)
        self.ag3 = GFFRM(128, 256, 3)
        self.d3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            MFIM(128, 64),
        )
        self.up2 = Up_(64, 64)
        self.ag2 = GFFRM(64, 128, 3)
        self.d2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            MFIM(128, 64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        up3 = self.up3(x3)
        ag3 = self.ag3(x2, up3)
        d3 = self.d3(ag3)
        up2 = self.up2(d3)
        ag2 = self.ag2(x1, up2)
        d2 = self.d2(ag2)
        return d2
if __name__ == '__main__':
    import time

    a = GURNet().cuda()
    b = torch.ones((1, 1, 128, 128)).cuda()
    out = a(b)
    print('out.shape = ', out.shape)
    print(summary(a, (1, 128, 128)))


