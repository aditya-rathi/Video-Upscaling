from main import *
from psnr import *
from torch.nn import functional
from torchvision.transforms import functional as vf
from PIL import Image
from math import log10, sqrt
import numpy as np

model = CNN(2)
model.load_state_dict = (torch.load('Final Final/400/400_epoch_checkpoint.pt'))
model.cuda()
for i in range(9):
    old_img = Image.open(f'Set14/{i}.png')
    width,height = old_img.size
    img = old_img.resize((width//2,height//2))
    img = img.convert('YCbCr')
    img,lr_cb,lr_cr = img.split()
    img = vf.to_tensor(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():
        outputs = model(img)
    outputs = torch.squeeze(outputs.cpu())
    outputs = outputs.numpy()
    outputs *=255.0
    outputs = outputs.clip(0,255)
    outputs = Image.fromarray(np.uint8(outputs),mode='L')
    lr_cb = lr_cb.resize((width,height),Image.BICUBIC)
    lr_cr = lr_cr.resize((width,height),Image.BICUBIC)
    output = Image.merge("YCbCr",(outputs,lr_cb,lr_cr)).convert("RGB")
    mse = np.mean((np.asarray(old_img) - np.asarray(output)) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print(f'{i}: psnr = ',psnr)

