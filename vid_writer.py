import ffmpeg

(ffmpeg
    .input('/home/aditya/Documents/Sem 2/Deep Learning/Video-Upscaling/video_synla/%d_output.png',r=30)
    .output('/home/aditya/Documents/Sem 2/Deep Learning/Video-Upscaling/output.mp4',pix_fmt='yuv420p')
    .run()
)
