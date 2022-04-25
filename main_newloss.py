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
from torchinfo import summary
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as Fv
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


train_on_gpu = torch.cuda.is_available()

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self, feature_model_extractor_node: str,
                 feature_model_normalize_mean: list,
                 feature_model_normalize_std: list) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        content_loss = functional.mse_loss(sr_feature, hr_feature)

        return content_loss

class TrainDataset(Dataset):
    def __init__(self,dataset_dir,lr_size,hr_size):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size,interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
            ])
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size,interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
            ])
    
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = hr_image.convert("YCbCr")
        hr_image,hr_cb,hr_cr = hr_image.split()
        hr_cb = hr_cb.resize((256,256),Image.BICUBIC)
        hr_cr = hr_cr.resize((256,256),Image.BICUBIC)
        hr_image = self.hr_transform(hr_image)
        lr_image = Image.open(self.image_filenames[index])
        lr_image = lr_image.convert('YCbCr')
        lr_image,lr_cb,lr_cr = lr_image.split()
        lr_cb = lr_cb.resize((256,256),Image.BICUBIC)
        lr_cr = lr_cr.resize((256,256),Image.BICUBIC)
        lr_image = self.lr_transform(lr_image)
        return lr_image, Fv.to_tensor(lr_cb), Fv.to_tensor(lr_cr), hr_image, Fv.to_tensor(hr_cb), Fv.to_tensor(hr_cr)

    def __len__(self):
        return len(self.image_filenames)

class CReLU(nn.Module):

    def __init__(self, inplace=False):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x,-x),1)
        return functional.relu(x)

class CNN(nn.Module):  
    def __init__(self,upscale_factor):
        super(CNN, self).__init__()
        self.crelu = CReLU()
        self.block_depth = 7
        self.mid_depth = 4
        self.upscale_factor = upscale_factor
        self.conv_layer = nn.Conv2d(1,self.mid_depth*2,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_block=nn.ModuleList()
        for i in range(self.block_depth):
            self.conv_block.append(nn.Sequential(nn.Conv2d(self.mid_depth*2,self.mid_depth,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        CReLU()))
        self.upscaler = nn.Conv2d(self.mid_depth*self.block_depth*2,1 * (self.upscale_factor ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x):
        # conv layers
        old_x = x
        depth_list = []
        x = self.conv_layer(x)
        for i in range(self.block_depth):
            
            x = self.conv_block[i](x)
            
            depth_list.append(x)
        x = torch.cat(depth_list,axis=1)
        x = self.upscaler(x)
        outputs = functional.pixel_shuffle(x, self.upscale_factor)
        outputs = outputs+functional.interpolate(old_x,scale_factor=2,mode='bilinear')
        return outputs

class Image_Upscaler():
    def __init__(self, dataset_dir, validation_dir, out_dir, lr_size, upscale_factor, batch_size, criterion, model):
        self.dataset_dir = dataset_dir
        self.validation_dir = validation_dir
        self.out_dir = out_dir
        self.criterion = criterion
        self.model = model(upscale_factor)
        self.lr_size = (lr_size[1],lr_size[0])
        self.hr_size = tuple(x*upscale_factor for x in self.lr_size)
        self.col_size = tuple(x*upscale_factor for x in reversed(self.lr_size))
        self.batch_size = batch_size
        self.trainset = TrainDataset(dataset_dir,self.lr_size,self.hr_size)
        self.trainloader = DataLoader(self.trainset,shuffle=True,batch_size=self.batch_size,num_workers=12)
        self.valset = TrainDataset(validation_dir,self.lr_size,self.hr_size)
        self.valloader = DataLoader(self.valset,shuffle=True,batch_size=self.batch_size,num_workers=12)
        self.train_on_gpu = torch.cuda.is_available()
        
    def upscale_image(self,epoch,image=None,fname=None):
        if image is None:
            img = Image.open(self.dataset_dir+fname)
            lr_transform = transforms.Resize(self.lr_size,interpolation=InterpolationMode.BICUBIC) #Resize takes input as height,width
            lr_image = lr_transform(img)
        elif fname is None:
            img = Image.fromarray(image)
            lr_image = img
        hr_transform = transforms.Resize(self.hr_size,interpolation=InterpolationMode.BICUBIC)
        tensorize = transforms.ToTensor()
        
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
        ground_name = self.out_dir+str(epoch)+'_ground.png'
        output_name = self.out_dir+str(epoch)+'_output.png'
        hr_image.save(ground_name)
        output.save(output_name)

    def train_mod(self):
        if train_on_gpu:
            self.model.cuda()
        optimizer = optim.RMSprop(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
        num_epoch = 200
        loss_list = []
        val_loss_list = []
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            print('Epoch = %2d'%epoch)
            optimizer.zero_grad()   # zero the parameter gradients
            loss_per_pixel = 0
            for _, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, lr_cb, lr_cr, labels, hr_cb, hr_cr = data
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = torch.cat([outputs,lr_cb,lr_cr],dim=1)
                labels = torch.cat([labels,hr_cb,hr_cr],dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_per_pixel += loss.item()/(self.hr_size[0]*self.hr_size[1]*self.batch_size)
            loss_per_pixel/=len(self.trainset)
            print(f'Train Loss = {loss_per_pixel:.5f}')
            scheduler.step(loss_per_pixel)
            loss_list.append(loss_per_pixel)

            #Visualize every 20 epochs
            if (epoch+1)%20==0:
                self.upscale_image(epoch,None,random.choice(listdir(self.dataset_dir)))

        print('Finished Training')
        torch.save(self.model.state_dict(),'model.pt')

        #Plot loss vs epochs
        loss_plotter(loss_list,val_loss_list)
        
        np.save('training_loss',np.array(loss_list))
    
    def load_checkpoint(self,path):
        self.model.load_state_dict = (torch.load(path))
        self.model.cuda()
    

if __name__ == "__main__":
    video = VideoReader("test2-360.mp4")
    dataset_dir = 'Synla-4096/'
    validation_dir = 'Synla-1024/'
    upscale_factor = 2
    batch_size = 128
    feature_model_extractor_node = "features.35"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]
    criterion = ContentLoss(feature_model_extractor_node,feature_model_normalize_mean,feature_model_normalize_std)
    model = CNN
    test_mode = 0

    lr_size = (128,128) if test_mode==0 else (video.width,video.height)
    out_dir = 'amv-test/'

    upscaler = Image_Upscaler(dataset_dir, validation_dir, out_dir, lr_size, upscale_factor, batch_size, criterion, model)

    if test_mode==1:
        upscaler.load_checkpoint("model.pt")
        for frame_idx in range(video.num_frames):
            #print(frame_idx)
            old_time = time.time()
            frame = video.get_frame()
            frame_upscaled = upscaler.upscale_image(frame_idx,frame)
            print(f"time taken = {time.time()-old_time:.3f}")
        video.complete()
        final_vid = VideoWriter(out_dir+'/%d_output.png','/amv_upscaled.mp4')
        final_vid.write_vid()
        
    else:
        #upscaler.load_checkpoint("model.pt")
        upscaler.train_mod()
