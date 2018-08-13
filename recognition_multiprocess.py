import cv2
import os
import numpy as np
import argparse
import time
from random import randint

from queue import Empty
from multiprocessing import Process, Queue, Event

import PyOpenPose

from nimble import Nimble, ClassifierParameters, HardNegativeParametersSample
from nimble import FeaturesExtractor, FeaturesParameters
from nimble import adapter
from nimble.dataset import ntu
from nimble import utils

radius = 4
line_width = 3
edges = [(0,1), (0,14), (14,16), (0,15), (15,17), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (1,11), (8,9), (9,10), (11,12), (12,13)]

#mapping = { \
#           0:'Standing still',
#            1:'Drinking',
#            2:'Clapping',
#            3:'Waving',
#            4:'Jumping',
#            5:'Bowing'
#          }

mapping = { \
            0:'Standing still',
            1:'Drinking',
            2:'Sitting',
            3:'Standing up',
            4:'Clapping',
            5:'Waving',
            6:'Jumping',
            7:'Phoning',
            8:'Bowing',
            9:'Falling'
          }


def draw_skeletons(skeletons, img):

    height, width = img.shape[:2]
    hip_centers = (skeletons[:,8,0] + skeletons[:,11,0])/2
    idx_center  = np.argmin(np.absolute(hip_centers - width / 2))

    for idx, skeleton in enumerate(skeletons):
        joints = np.ravel(skeleton)
        if idx == idx_center:
            color = (0,255,0)
        else:
            color = (0,0,255)
        draw_sticks(joints, color, img)
        draw_joints(joints, color, img)

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

def draw_action(action, img):
    cv2.putText(img, action, (5,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)

def draw_fps(timestep, img):
    height, width = img.shape[:2]

    fps = 1 / timestep
    location_y = int(height - 5)
    location_x = int(width - width/3.5)
    cv2.putText(img, '{:.1f} fps'.format(fps), (location_x, location_y), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)

def clear_queue(queue):
    try:
        queue.get_nowait()
    except Empty:
        pass

def skeleton_loop(queue, stop, openpose_dir):
    # Init Openpose
    op = PyOpenPose.OpenPose((240,320), (368,368), (1280,720), 'COCO', openpose_dir + os.sep, 0, False)

    # Init camera
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320.0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

    while not stop.is_set():
        # Read new image and apply transformation
        ret, frame = cap.read()
        if frame is None:
            print('Could not capture camera')
            stop.set()
            break

        frame = cv2.flip(cv2.transpose(frame), flipCode = 1) # 90 clockwise rotation

        # Detect pose
        op.detectPose(frame)
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        clear_queue(queue)
        queue.put((frame,persons))

    cap.release()
    print('Quitting skeleton extraction')

def gesture_recognition_loop(skeleton_q, gesture_q, stop, threshold, model_file):
    (model, extractor) = utils.load(model_file)


    before = time.perf_counter()
    while not stop.is_set():
        try:
            (frame, persons) = skeleton_q.get(timeout = 1)
        except Empty:
            if stop.is_set():
                break
            continue

        if persons is None:
            before = time.perf_counter()

            clear_queue(gesture_q)
            gesture_q.put((frame, None, None))
            continue

        height, width = frame.shape[:2]
        hip_centers = (persons[:,8,0] + persons[:,11,0])/2
        idx_center  = np.argmin(np.absolute(hip_centers - width / 2))

        person = np.ravel(persons[idx_center])

        now = time.perf_counter()
        timestep = now - before
        before = now

        descriptor = extractor.extract_descriptor_online(person, timestep, adapter.openpose_adapter, 0.3)
       	if descriptor is None:
            clear_queue(gesture_q)
            gesture_q.put((frame, persons, None))
            continue

        descriptor = np.expand_dims(descriptor, axis = 0)
        (_,_,_,_,gesture,score) = model.predict(descriptor, detection = True)

        gesture_t = gesture[0]
        if gesture_t != -1 and score[gesture_t] < threshold:
                print('Detected action {} below threshold'.format(mapping[gesture_t]))
                gesture_t = -1

        clear_queue(gesture_q)
        gesture_q.put((frame, persons, gesture_t,))

    print('Quitting gesture recognition')

def display_image(stop, queue):

    detected_action = -1
    before = time.perf_counter()
    while not stop.is_set():
        try:
            (frame, persons, results) = queue.get(timeout = 1)
        except Empty:
            if stop.is_set():
                break
            continue

        if persons is not None:
            draw_skeletons(persons, frame)

        if results is not None:
            if results != -1:
                detected_action = results
                display_action_t = time.perf_counter()
                draw_action(mapping[detected_action], frame)
            elif detected_action != -1 and time.perf_counter() - display_action_t < 3.0:
                draw_action(mapping[detected_action], frame)

        now = time.perf_counter()
        timestep = now - before
        before = now
        draw_fps(timestep, frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    print('Quitting displaying_image')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpose_dir", type = str, required = True)
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--threshold", type = float, default = 0.9)
    args = parser.parse_args()

    model_file = args.model
    openpose_dir = args.openpose_dir
    if not os.path.isdir(openpose_dir):
        raise NotADirectoryError('openpose_dir must point to a directory')

    skeleton_q = Queue(maxsize = 1)
    gesture_q  = Queue(maxsize = 1)
    stop = Event()

    t1 = Process(target = skeleton_loop, args=(skeleton_q, stop, openpose_dir)); t1.start()
    t2 = Process(target = gesture_recognition_loop, args=(skeleton_q, gesture_q, stop, args.threshold, model_file)); t2.start()

    display_image(stop, gesture_q)
    stop.set()

    # clear items in the queue to allow joining process
    clear_queue(skeleton_q)
    clear_queue(gesture_q)

    t1.join()
    t2.join()
