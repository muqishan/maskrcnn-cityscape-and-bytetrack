'''
video 1.mp4 2.mp4 and 3.mp4
train and valid in 1.ma4 2.mp4  frames % 8
test in 3.mp4
Instance segmentation tasks can be both segmentation and detection tasks
Use labelme to label in imgs

'''
import cv2
import os
import time

video_path = ['datasets/video/3.mp4']
BASE_DIR = os.getcwd()
img_path = 'datasets/imgs'  # marked
os.path.exists(img_path) or os.mkdir(img_path)
idx = -1
year = time.localtime(time.time())[0]
for path in video_path:
    capture = cv2.VideoCapture(path)
    print(path, ' PNGImages:', capture.get(7))
    while True:
        status, frame = capture.read()
        if status:
            if idx % 7 == 0:
                img = cv2.resize(frame, (256, 256))
                index = '0'*(4-len(str(idx))) + str(idx)
                cv2.imwrite(os.path.join(img_path, 'frame_'+index+'.png'), img)
            idx += 1
            if idx ==100:
                break
        else:
            break

