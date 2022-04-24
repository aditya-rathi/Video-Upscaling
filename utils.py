import cv2
import numpy as np
import ffmpeg
import os

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


        

# class VideoWriter():
#     def __init__(self,path):
#         self.path = path
         
#     def compile_frames(self):
#         img_array = []
#         for filename in sorted(glob.glob(self.path)):
#             print(filename)
#             img = cv2.imread(filename)
#             height, width, layers = img.shape
#             size = (width,height)
#             img_array.append(img)
    
    
#         out = cv2.VideoWriter('/home/aditya/Documents/CMU Sem 2/Deep-Learning/Video-Upscaling/project.avi',cv2.VideoWriter_fourcc(*'X264'), 30, size)
        
#         for i in range(len(img_array)):
#             out.write(img_array[i])
#         out.release()
