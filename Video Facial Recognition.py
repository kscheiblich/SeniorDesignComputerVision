# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:25:00 2021

@author: sirk2
"""

import face_alignment
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import collections
import time

cap = cv2.VideoCapture("KrisVid.mp4")
frames = []
while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

print(frames)


# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

# Run the 3D face alignment on a test video, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')

t_start = time.time()
preds = fa.get_landmarks_from_image(frames[0])
print(f'BlazeFace: Execution time for a single image: {time.time() - t_start}')

#BlazeFace: Execution time for a single image: 1.6376028060913086

plt.imshow(frames[0])
for detection in preds:
    plt.scatter(detection[:,0], detection[:,1], 2)