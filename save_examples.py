import glob
import torch
import dataset
import numpy as np
from unet import UNet
import torch.nn as nn
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

if __name__ == "__main__":

    val_list = glob.glob('/content/test/*png')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    
    ll = []
    with torch.no_grad():
        for i,(img,gt) in enumerate(val_loader):
          if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
          img, gt = Variable(img), Variable(gt)
          gt = gt.clamp(min = 0, max = 1)
          gt_np = gt.squeeze().cpu().detach().numpy()
          plt.imsave(f"/content/knowledge-distillation-for-unet/examples4/gt_{i}.png", gt_np, cmap='gray')

          for channel_depth, weight in [(1, 'CP_1_1.pth'), (4, 'CP_4_4.pth'), (4, 'CP_4_student5.pth'),(16, 'CP_16_4.pth'), (16, 'CP_16_student5.pth'), (32, 'CP_32_5.pth')]:
            teacher = UNet(channel_depth = channel_depth, n_channels = 3, n_classes=1)
            teacher.load_state_dict(torch.load('/content/' + weight))
            teacher.eval().cuda()
            output = teacher(img)
            output = output.clamp(min = 0, max = 1)
            output_np = output.squeeze().cpu().detach().numpy()
            plt.imsave(f"/content/knowledge-distillation-for-unet/examples4/{weight}_output_{i}.png", output_np, cmap='gray')
          break
