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
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns


def plot_latency(data):

    # Calculate mean and variance
    mean = np.mean(data)
    variance = np.var(data)

    # Create a KDE distribution plot
    sns.set_style('whitegrid')  # Optional: Set the plot style
    sns.kdeplot(data, shade=True, color='blue')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Distribution Plot for Continuous Values')

    # Optionally, you can add a rug plot to show individual data points
    sns.rugplot(data, color='red', alpha=0.5)

    # Add lines for mean and variance
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + np.sqrt(variance), color='g', linestyle='dashed', linewidth=2, label=f'Std Dev: {np.sqrt(variance):.2f}')

    # Save the plot as an image (e.g., PNG or PDF)
    plt.savefig('continuous_distribution_plot.png')  # Change the file format as needed

    # Display the plot with a legend
    plt.legend()
    plt.show()


    # plt.savefig('distribution_plot.png')
    # Display the plot
    # plt.show()


def evaluate(teacher, val_loader):
    teacher.eval().cuda()
    data = []

    criterion = nn.BCEWithLogitsLoss()
    ll = []
    with torch.no_grad():
        for i,(img,gt) in tqdm(enumerate(val_loader), total=len(val_loader)):
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
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)
            ll.append(loss.item())
            data.append(seconds)

    
    mean = np.mean(data)
    variance = np.var(data)

    mean_dice = np.mean(ll)
    print('Test metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))
    print(mean)
    print(variance)
    # plot_latency(data)


if __name__ == "__main__":

    teacher = UNet(channel_depth = 1, n_channels = 3, n_classes=1)
    teacher.load_state_dict(torch.load("/home/sathya/student_cp/unet_8_1_128_maxpool_1e4_lr/CP_1_student20.pth"))
    test_list = glob.glob('/home/sathya/test/*png')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    test = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    

    evaluate(teacher, test)

