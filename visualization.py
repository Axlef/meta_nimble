import cv2
import os
import numpy as np
import argparse
import time
import imutils

import PyOpenPose


radius = 4
line_width = 3
edges = [(0,1), (0,14), (14,16), (0,15), (15,17), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (1,11), (8,9), (9,10), (11,12), (12,13)]

def draw_sticks(joints, edges, color, img):
    for edge1, edge2 in edges:
        joint1 = joints[edge1]
        joint2 = joints[edge2]
        if joint1[2] >= 0.3 and joint2[2] >= 0.3:
            joint1 = tuple(map(int, joints[edge1][0:2]))
            joint2 = tuple(map(int, joints[edge2][0:2]))
            cv2.line(img, joint1, joint2, color, line_width)

def draw_joints(joints, color, img):
    for joint in joints:
        if joint[2] >= 0.3:
            center = tuple(map(int, joint[0:2]))
            cv2.circle(img, center, radius, color, -1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpose_dir", type = str, required = True)

    args = parser.parse_args()
    openpose_dir = args.openpose_dir
    if not os.path.isdir(openpose_dir):
        raise NotADirectoryError('openpose_dir must point to a directory')

    ### Openpose ###
    op = PyOpenPose.OpenPose((240,320), (368,368), (1280,720), 'COCO', openpose_dir + os.sep,  0, False)

    ### Webcam ###
    cap = cv2.VideoCapture(0)

    while True:
        # Read new image and apply transformation
        ret, frame = cap.read()
        frame = imutils.rotate_bound(frame, angle = 90)

        # Detect pose
        op.detectPose(frame)
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        if persons is None:
            cv2.imshow('frame', frame)
            if cv2.waitKey(10) == 27:
                break
            continue

        persons_list = persons.tolist()
        color = (255,0,0)
        for person in persons_list:
            draw_sticks(person, edges, color, frame)
            draw_joints(person, color, frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
