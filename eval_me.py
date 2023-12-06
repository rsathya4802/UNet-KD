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
import time

if __name__ == "__main__":

    teacher = UNet(channel_depth = 1, n_channels = 3, n_classes=1)
    # teacher.load_state_dict(torch.load("/home/sathya/teacher_cp/unet_4_256_maxpool/CP_4_10.pth"))  
    # teacher.load_state_dict(torch.load("/home/sathya/teacher_cp/unet_1_128_maxpool/CP_1_10.pth"))
    teacher.load_state_dict(torch.load("/home/sathya/student_cp/unet_8_1_128_maxpool/CP_1_student10.pth"))
    teacher.eval().cuda()
    val_list = glob.glob('/home/sathya/val/*png')

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

            start_time= time.time() # set the time at which inference started
            output = teacher(img)
            stop_time=time.time()
            duration =stop_time - start_time
            hours = duration // 3600
            minutes = (duration - (hours * 3600)) // 60
            seconds = duration - ((hours * 3600) + (minutes * 60))
            msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.4f} seconds)'
            print (msg, flush=True) # print out inferenceduration time
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)

            output_np = output.squeeze().cpu().detach().numpy()
            gt_np = gt.squeeze().cpu().detach().numpy()
            # img = np.transpose(img.squeeze(0).cpu().detach().numpy(), (1,2,0))
            # print(img.shape)
            # plt.imsave(f"input_{i}.png", img, cmap='gray')
            plt.imsave(f"output_{i}.png", output_np, cmap='gray')
            plt.imsave(f"gt_{i}.png", gt_np, cmap='gray')
            break
