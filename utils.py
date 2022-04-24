import cv2
import numpy as np
import ffmpeg
import os
import matplotlib.pyplot as plt

class VideoReader():
    def __init__(self, path):
        self.vid = cv2.VideoCapture(path)
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_frame(self):
        if self.vid.isOpened():
            success, frame_bgr = self.vid.read()
        if success:
            frame_bgr = np.array(frame_bgr)
            frame_rgb = frame_bgr[:, :, ::-1]
            return frame_rgb
        else:
            return None

    def complete(self):
        self.vid.release()

class VideoWriter():
    def __init__(self,path,output):
        self.dir = path
        self.out_path = output
    
    def write_vid(self):
        (ffmpeg
            .input(self.dir,r=30)
            .output(os.getcwd()+self.out_path,pix_fmt='yuv420p',preset = 'veryslow',tune = 'animation')
            .run()
        )    


def loss_plotter(loss_list,val_loss_list):
    plt.plot(range(1,len(loss_list)+1),loss_list,range(1,len(loss_list)+1),val_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss Per Pixel')
    plt.title('Loss per pixel vs Epochs')
    plt.legend('Training Loss','Validation Loss')
    plt.show()
