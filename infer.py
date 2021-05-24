from model import unet
from data import label_visualize
import cv2
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

number_of_class = 4
test_img_count = 299
epochs = 15
batch_size = 16
network_size = (512, 512)

Obstacles = [225, 109, 50]
Water = [10, 100, 152]
Sky = [223, 218, 214]
Ignore = [0, 0, 0]
COLOR_DICT = np.array([Water, Obstacles, Sky, Ignore])

# Read the video from specified path
cap = cv2.VideoCapture("/home/bbayraktar/cut.mp4")

model = unet(input_size=network_size + (3,), num_class=number_of_class,  pretrained_weights=None)
model.load_weights("unet_mastr-512.hdf5")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, 30.0, (1280, 720))

while True:
    ret, img = cap.read()
    if ret:
        
        img = img / 255
        # resize img to network size
        img = cv2.resize(img, network_size)

        predicted = model.predict(img[np.newaxis, ...])
        predicted = predicted[0]

        predicted = label_visualize(number_of_class, COLOR_DICT, predicted)

        blended = cv2.addWeighted(np.float32(img), 0.5, np.float32(predicted), 0.5, 0)
        cv2.imshow('infer', blended)
        cv2.waitKey(1)

cap.release()

# Release all space and windows once done
cap.release()
cv2.destroyAllWindows()




