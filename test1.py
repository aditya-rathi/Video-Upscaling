from cmath import tanh
from matplotlib.image import imsave
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
import matplotlib.pyplot as plt
import random

train_on_gpu = torch.cuda.is_available()

class TrainDataset(Dataset):
    def __init__(self,dataset_dir):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
        self.lr_transform = transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor()
            ])
        self.hr_transform = transforms.Compose([
            transforms.Resize((200,200)),
            transforms.ToTensor()
            ])
    
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = hr_image.convert("YCbCr")
        hr_image = self.hr_transform(hr_image)
        lr_image = Image.open(self.image_filenames[index])
        lr_image = lr_image.convert('YCbCr')
        lr_image = self.lr_transform(lr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(128, 1 * (2 ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
    
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        outputs = functional.pixel_shuffle(x, 2)
        return outputs

def upscale_image(epoch,fname,model):
    img = Image.open('train_data/291/'+fname)
    lr_transform = transforms.Resize((100,100))
    hr_transform = transforms.Resize((200,200))
    tensorize = transforms.ToTensor()
    lr_image = lr_transform(img)
    lr_image = lr_image.convert("YCbCr")
    lr_y,lr_cb,lr_cr = lr_image.split()
    hr_image = hr_transform(img)
    lr_y = tensorize(lr_y)
    with torch.no_grad():
        output = model(torch.unsqueeze(lr_y,1).cuda())
    output = output[0,0].cpu().numpy()
    output *=255.0
    output = output.clip(0,255)
    output = Image.fromarray(np.uint8(output),mode="L")
    lr_cb = lr_cb.resize((200,200),Image.BICUBIC)
    lr_cr = lr_cr.resize((200,200),Image.BICUBIC)
    output = Image.merge("YCbCr",(output,lr_cb,lr_cr)).convert("RGB")
    ground_name = 'data/'+str(epoch)+'_ground.png'
    output_name = 'data/'+str(epoch)+'_output.png'
    hr_image.save(ground_name)
    output.save(output_name)

def train_mod(model,criterion,trainloader):
    if train_on_gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
    num_epoch = 200
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        print('Epoch = %2d'%epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            lr_image,hr_image = data
            inputs, labels = torch.unsqueeze(lr_image[:,0,:,:],1),torch.unsqueeze(hr_image[:,0,:,:],1)
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(loss.item())    
        scheduler.step(loss.item())

        upscale_image(epoch,random.choice(listdir('train_data/291/')),model)

        # ground_name = 'data/'+str(epoch)+'_ground.png'
        # output_name = 'data/'+str(epoch)+'_output.png'
        # o_img = ((outputs[0,0,:,:].detach()).cpu()).numpy()
        # o_img*=255
        # o_img = np.clip(o_img,0,255)
        # o_img = np.uint8(o_img)
        
        # plt.imsave(ground_name,labels[0,0,:,:].detach(),cmap='gray')
        # plt.imsave(output_name,o_img,cmap='gray')

    print('Finished Training')
    torch.save(model.state_dict(),'model.pt')

def data_loader():
    trainset = TrainDataset('train_data/291/')
    trainloader = DataLoader(trainset,shuffle=True,batch_size=12)
    criterion = nn.MSELoss()
    model = CNN()
    return model,criterion,trainloader

def main():

    model,criterion,trainloader = data_loader()    
    train_mod(model,criterion,trainloader)
    

if __name__ == "__main__":
    main()