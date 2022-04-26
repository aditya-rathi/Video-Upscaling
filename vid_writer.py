import ffmpeg
import os

(ffmpeg
    .input(os.getcwd()+'/amv-test/%d_output.png',r=30)
    .output(os.getcwd()+'/amv-output.mp4',pix_fmt='yuv420p',preset='veryslow',tune='animation')
    .run()
)
