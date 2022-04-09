import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from os.path import join
from os import listdir
from PIL import Image
from torch.nn import functional
from utils import *
import random
import time

train_on_gpu = torch.cuda.is_available()

class TrainDataset(Dataset):
    def __init__(self,dataset_dir,lr_size,hr_size):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size),
            transforms.ToTensor()
            ])
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size),
            transforms.ToTensor()
            ])
    
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = hr_image.convert("YCbCr")
        hr_image,_,_ = hr_image.split()
        hr_image = self.hr_transform(hr_image)
        lr_image = Image.open(self.image_filenames[index])
        lr_image = lr_image.convert('YCbCr')
        lr_image,_,_ = lr_image.split()
        lr_image = self.lr_transform(lr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class CNN(nn.Module):  
    def __init__(self,upscale_factor):
        super(CNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv_layer = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(128, 1 * (self.upscale_factor ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        
    
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        outputs = functional.pixel_shuffle(x, self.upscale_factor)
        return outputs

class Image_Upscaler():
    def __init__(self, dataset_dir, lr_size, upscale_factor, batch_size, criterion, model):
        self.dataset_dir = dataset_dir
        self.criterion = criterion
        self.model = model(upscale_factor)
        self.lr_size = (lr_size[1],lr_size[0])
        self.hr_size = tuple(x*upscale_factor for x in self.lr_size)
        self.col_size = tuple(x*upscale_factor for x in reversed(self.lr_size))
        self.trainset = TrainDataset(dataset_dir,self.lr_size,self.hr_size)
        self.trainloader = DataLoader(self.trainset,shuffle=True,batch_size=batch_size)
        self.train_on_gpu = torch.cuda.is_available()
        
    def upscale_image(self,epoch,image=None,fname=None):
        if image is None:
            img = Image.open(self.dataset_dir+fname)
        elif fname is None:
            img = Image.fromarray(image)
        lr_transform = transforms.Resize(self.lr_size) #Resize takes input as height,width
        hr_transform = transforms.Resize(self.hr_size)
        tensorize = transforms.ToTensor()
        lr_image = lr_transform(img)
        lr_image = lr_image.convert("YCbCr")
        lr_y,lr_cb,lr_cr = lr_image.split()
        hr_image = hr_transform(img)
        lr_y = tensorize(lr_y)
        with torch.no_grad():
            output = self.model(torch.unsqueeze(lr_y,1).cuda())
        if train_on_gpu:
            output = output[0,0].cpu().numpy()
        else:
            output = output[0,0].numpy()
        output *=255.0
        output = output.clip(0,255)
        output = Image.fromarray(np.uint8(output),mode="L")
        lr_cb = lr_cb.resize(self.col_size,Image.BICUBIC)
        lr_cr = lr_cr.resize(self.col_size,Image.BICUBIC)
        output = Image.merge("YCbCr",(output,lr_cb,lr_cr)).convert("RGB")
        ground_name = 'output/'+str(epoch)+'_ground.png'
        output_name = 'temp/'+str(epoch)+'_output.png'
        #hr_image.save(ground_name)
        output.save(output_name)

    def train_mod(self):
        if train_on_gpu:
            self.model.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
        num_epoch = 200
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            print('Epoch = %2d'%epoch)
            for _, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(loss.item())    
            scheduler.step(loss.item())

            self.upscale_image(epoch,None,random.choice(listdir(self.dataset_dir)))


        print('Finished Training')
        torch.save(self.model.state_dict(),'model.pt')
    
    def load_checkpoint(self,path):
        self.model.load_state_dict = (torch.load(path))
        self.model.cuda()
    

if __name__ == "__main__":
    video = VideoReader("test_vid_360.mp4")
    dataset_dir = 'train2/'
    lr_size = (video.width,video.height)
    #lr_size = (102,100)
    upscale_factor = 2
    batch_size = 12
    criterion = nn.MSELoss()
    model = CNN
    test_mode = 1

    upscaler = Image_Upscaler(dataset_dir, lr_size, upscale_factor, batch_size, criterion, model)

    if test_mode==1:
        upscaler.load_checkpoint("model.pt")
        for frame_idx in range(video.num_frames):
            #print(frame_idx)
            old_time = time.time()
            frame = video.get_frame()
            frame_upscaled = upscaler.upscale_image(frame_idx,frame)
            print(f"time taken = {time.time()-old_time:.2f}")
        video.complete()
        

    else:
        upscaler.load_checkpoint("model.pt")
        upscaler.train_mod()