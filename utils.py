import cv2
import numpy as np

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

