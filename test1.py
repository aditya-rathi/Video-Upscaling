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

train_on_gpu = torch.cuda.is_available()

# Change for PSNR
def accuracy(model,testloader):
    correct = 0 
    total = 0
    for data in testloader:
            images, labels = data
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('acc = %.3f' %(correct/total))

class TrainDataset(Dataset):
    def __init__(self,dataset_dir):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((100,100)),
            transforms.ToTensor()])
        self.hr_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((200,200)),
            transforms.ToTensor()])
    
    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.image_filenames)

class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(128, 1 * (2 ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
    
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        outputs = functional.pixel_shuffle(x, 2)

        return outputs

def main():
    
    trainset = TrainDataset('train_data/291/')
    trainloader = DataLoader(trainset,shuffle=True,batch_size=3)
    criterion = nn.MSELoss()
    model = CNN()

    if train_on_gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    num_epoch = 200
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        print('Epoch = %2d'%epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        ground_name = 'data/'+str(epoch)+'_ground.png'
        output_name = 'data/'+str(epoch)+'_output.png'
        o_img = outputs[0,0,:,:].detach()
        #o_img*=255
        plt.imsave(ground_name,labels[0,0,:,:].detach(),cmap='gray')
        plt.imsave(output_name,o_img,cmap='gray')

    print('Finished Training')

if __name__ == "__main__":
    main()