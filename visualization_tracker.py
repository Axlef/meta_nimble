import cv2
import os
import numpy as np
import argparse
import time
import imutils

import PyOpenPose

from tracker.tracker import PoseTracker
import tracker.skeleton_utils as utils


radius = 4
line_width = 3
edges = [(0,1), (0,14), (14,16), (0,15), (15,17), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (1,11), (8,9), (9,10), (11,12), (12,13)]
colors = ( (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255),(192,192,192), (128,128,128),  (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128), (0,0,128) )

def draw_instances(instances, img):
    for id, pose in instances.items():
        color = colors[id%len(colors)]
        draw_sticks(pose, color, img)
        draw_joints(pose, color, img)

        bb = utils.extract_bounding_box(pose, 0.3)
        if bb is not None:
            text_location = (int((bb[0] + bb[2])/2) - 20, int(bb[1] - 20))
            draw_bounding_box(bb, color, img)
            draw_id(id, text_location, img)

def draw_sticks(joints, color, img):
    for edge1, edge2 in edges:
        confidence1 = joints[3*edge1+2]
        confidence2 = joints[3*edge2+2]
        if confidence1 >= 0.3 and confidence2 >= 0.3:
            joint1 = tuple(map(int, joints[3*edge1:3*edge1+2]))
            joint2 = tuple(map(int, joints[3*edge2:3*edge2+2]))
            cv2.line(img, joint1, joint2, color, line_width)

def draw_joints(joints, color, img):
    for idx in range(len(joints)//3):
        if joints[3*idx+2] >= 0.3:
            center = tuple(map(int, joints[3*idx:3*idx+2]))
            cv2.circle(img, center, radius, color, -1)

def draw_bounding_box(bb, color, img):
    xy_top_left = tuple(map(int, bb[0:2]))
    xy_bottom_right = tuple(map(int, bb[2:4]))
    cv2.rectangle(img, xy_top_left, xy_bottom_right, color, line_width)

def draw_id(id, location, img):
    cv2.putText(img, '[id: {}]'.format(id), location, cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpose_dir", type = str, required = True)

    args = parser.parse_args()
    openpose_dir = args.openpose_dir
    if not os.path.isdir(openpose_dir):
        raise NotADirectoryError('openpose_dir must point to a directory')

    ### Openpose ###
    op = PyOpenPose.OpenPose((240,320), (368,368), (1280,720), 'COCO', openpose_dir + os.sep + 'models' + os.sep,  0, False)

    ### Tracker ###
    tracker = PoseTracker()

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

        n_persons = persons.shape[0]
        persons = persons.reshape((n_persons, 3 * 18))

        tracker.update(persons.tolist())
        instances = tracker.get_current_poses()

        draw_instances(instances, frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
