# -*- coding: utf-8 -*-
# All rights reserved by Vincent Bouget, Arthur Filoche, Anastase Charantonis, Dominique Béréziat, Julien Brajard
# A research work funded by Sorbonne Center for Artificial Intelligence (Sorbonne Université)

from unet_parts import *

class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear=True, n=8):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # Change n to change weights

        self.inc = DoubleConv(n_channels, 2*n)
        self.down1 = Down(2*n, 4*n)
        self.down2 = Down(4*n, 8*n)
        self.down3 = Down(8*n, 16*n)
        factor = 2 if bilinear else 1
        self.down4 = Down(16*n, 32*n // factor)
        self.up1 = Up(32*n, 16*n, bilinear)
        self.up2 = Up(16*n, 8*n, bilinear)
        self.up3 = Up(8*n, 4*n, bilinear)
        self.up4 = Up(4*n, 2*n*factor, bilinear)
        self.outc = OutConv(2*n, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def get_nb_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
