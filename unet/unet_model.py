# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, channel_depth, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, channel_depth)

        self.down1 = down(channel_depth, channel_depth)
        self.down2 = down(channel_depth, channel_depth*2)               
        self.down3 = down(channel_depth*2, channel_depth*2)
        self.down4 = down(channel_depth*2, channel_depth*4)
        self.down5 = down(channel_depth*4, channel_depth*4)
        self.down6 = down(channel_depth*4, channel_depth*4)
        
        self.up6 = up(channel_depth*8, channel_depth*4)
        self.up5 = up(channel_depth*8, channel_depth*4)
        self.up4 = up(channel_depth*6, channel_depth*2)
        self.up3 = up(channel_depth*4, channel_depth*2)
        self.up2 = up(channel_depth*3, channel_depth)
        self.up1 = up(channel_depth*2, channel_depth)
        self.outc = outconv(channel_depth, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        
        x = self.up6(x7, x6)
        x = self.up5(x, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)
        return x


'''
class UNet(nn.Module):
    def __init__(self, channel_depth, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, channel_depth)
        self.down1 = down(channel_depth, channel_depth*2)
        self.down2 = down(channel_depth*2, channel_depth*4)
        self.down3 = down(channel_depth*4, channel_depth*8)
        self.down4 = down(channel_depth*8, channel_depth*16)
        self.down5 = down(channel_depth*16, channel_depth*32)
        self.down6 = down(channel_depth*32, channel_depth*64)
        self.down7 = down(channel_depth*64, channel_depth*64)
        self.down8 = down(channel_depth*64, channel_depth*64)
        self.up1 = up(channel_depth*128, channel_depth*64)
        self.up2 = up(channel_depth*128, channel_depth*32)
        self.up3 = up(channel_depth*64, channel_depth*16)
        self.up4 = up(channel_depth*32, channel_depth*8)
        self.up5 = up(channel_depth*16, channel_depth*4)
        self.up6 = up(channel_depth*8, channel_depth*2)
        self.up7 = up(channel_depth*4, channel_depth)
        self.up8 = up(channel_depth*2, channel_depth)
        self.outc = outconv(channel_depth, n_classes)
        # self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        x = self.outc(x)
        return x


      # def encode(self,x)
'''

'''
class UNet(nn.Module):
    def __init__(self, channel_depth, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, channel_depth)
        self.down1 = down(channel_depth, channel_depth*2)
        self.down2 = down(channel_depth*2, channel_depth*4)
        self.down3 = down(channel_depth*4, channel_depth*8)
        self.down4 = down(channel_depth*8, channel_depth*8)
        self.up1 = up(channel_depth*16, channel_depth*4)
        self.up2 = up(channel_depth*8, channel_depth*2)
        self.up3 = up(channel_depth*4, channel_depth)
        self.up4 = up(channel_depth*2, channel_depth)
        self.outc = outconv(channel_depth, n_classes)

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
        x = self.outc(x)
        return x
        #return x

'''

'''class UNet16(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.up4 = up(32, 16)
        self.outc = outconv(16, n_classes)

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
        x = self.outc(x)
        return F.sigmoid(x)
        #return x
'''
