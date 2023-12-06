import glob
import torch
import dataset
import numpy as np
from utils import *
from unet import UNet
from loss import loss_fn_kd
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

teacher_weights = '/home/sathya/teacher_cp/unet_8_128_maxpool/CP_8_10.pth'
#student_weights = 'checkpoints/CP5.pth'
num_of_epochs = 40
summary_steps = 10

def train_student(student, teacher, optimizer, train_loader):
    print('-------Train student-------')
    #called once for each epoch
    student.train().cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = teacher.eval().cuda()

    summ = []
    for i, (img, gt) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if torch.cuda.is_available():
            img = img.cuda()  # move img to the device
            teacher_output = teacher(img)
            img, gt = img.cuda(), gt.cuda()
            teacher_output = teacher_output.cuda()
            torch.cuda.empty_cache()

        img, gt = Variable(img), Variable(gt)
        teacher_output =  Variable(teacher_output)

        output = student(img)

        #TODO: loss is wrong
        loss = loss_fn_kd(output, teacher_output, gt)    

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        if i % summary_steps == 0:
            #do i need to move it to CPU?
            gt = gt.clamp(min = 0, max = 1)
            output = output.clamp(min = 0, max = 1)            
            metric = dice_loss(output, gt)
            summary = {'metric' : metric.item(), 'loss' : loss.item()}
            summ.append(summary)
    
    #print('Average loss over this epoch: ' + np.mean(loss_avg))
    mean_dice_coeff =  np.mean([x['metric'] for x in summ])
    mean_loss = np.mean([x['loss'] for x in summ])
    print('- Train metrics:\n' + '\tMetric:{}\n\tLoss:{}'.format(mean_dice_coeff, mean_loss))
    #print accuracy and loss

def evaluate_kd(student, val_loader):
    print('-------Evaluate student-------')
    student.eval().cuda()

    #criterion = torch.nn.BCEWithLogitsLoss()
    loss_summ = []
    with torch.no_grad():
        for i, (img, gt) in enumerate(val_loader):
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)

            output = student(img)
            gt = gt.clamp(min = 0, max = 1)
            output = output.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)

            loss_summ.append(loss.item())

    mean_loss = np.mean(loss_summ)
    print('- Eval metrics:\n\tAverage Dice loss:{}'.format(mean_loss))
    return mean_loss

if __name__ == "__main__":
    min_loss = 100

    teacher = UNet(channel_depth = 8, n_channels = 3, n_classes=1)
    student = UNet(channel_depth = 1, n_channels = 3, n_classes=1)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

    #load teacher and student model
    teacher.load_state_dict(torch.load(teacher_weights))
    #student.load_state_dict(torch.load(student_weights))

    #NV: add val folder
    train_list = glob.glob('/home/sathya/train/*png')
    val_list = glob.glob('/home/sathya/val/*png')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #2 tensors -> img_list and gt_list. for batch_size = 1 --> img: (1, 3, 320, 320); gt: (1, 1, 320, 320)
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )


    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )

    #train_and_evaluate_kd:
    #get teacher outputs as list of tensors
    #teacher_outputs = fetch_teacher_outputs(teacher, train_loader)
    #print(len(teacher_outputs))
    for epoch in range(num_of_epochs):
        #train the student
        print(' --- student training: epoch {}'.format(epoch+1))
        train_student(student, teacher, optimizer, train_loader)

        #evaluate for one epoch on validation set
        val = evaluate_kd(student, val_loader)
        if(val < min_loss):
            min_loss = val
            #TODO: make min as the val loss of teacher
            print('New best!!')


        #if val_metric is best, add checkpoint

        torch.save(student.state_dict(), '/home/sathya/student_cp/unet_8_1_128_maxpool_40epochs/CP_1_student{}.pth'.format(epoch+1))
        print("Checkpoint {} saved!".format(epoch+1))
        scheduler.step()
        






