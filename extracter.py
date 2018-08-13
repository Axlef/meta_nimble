import cv2
import os
import argparse
import time
import imutils
import json
import glob

import numpy as np

import PyOpenPose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral_dir', type = str, required = True)
    parser.add_argument("--openpose_dir", type = str, required = True)
    parser.add_argument("--action", type = int, default = -1)
    args = parser.parse_args()

    neutral_dir = args.neutral_dir
    openpose_dir = args.openpose_dir
    action = args.action
    if not os.path.isdir(neutral_dir):
        raise IOError('neutral_dir must point to a directory')
    if not os.path.isdir(openpose_dir):
        raise NotADirectoryError('openpose_dir must point to a directory')

     ### Openpose ###
    op = PyOpenPose.OpenPose((240,320), (368,368), (1280,720), 'COCO', openpose_dir + os.sep + 'models' + os.sep, 0, False)

    for videofile in sorted(glob.glob(neutral_dir + '**/*.mp4', recursive=True)):
        print('Processing video: {}'.format(videofile))
        skeletonfile = os.path.splitext(videofile)[0] + '.skeleton'

        ### Webcam ###
        cap = cv2.VideoCapture(videofile)
        if not cap.isOpened():
            raise cv2.error('Unable to open video capture')
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start = time.perf_counter()
        frames = []
        counter = 0
        while(cap.isOpened()):
            # Read new image and apply transformation
            ret, frame = cap.read()
            if not ret:
                break

            op.detectPose(frame)
            persons = op.getKeypoints(op.KeypointType.POSE)[0]

            if persons is None:
                counter += 1
                continue
                # raise Exception('No person in the frame')

            person = np.ravel(persons[0])
            frame_annotations = {'pose':person.tolist(), 'id':counter}
            frames.append(frame_annotations)
            counter += 1

        annotations_txt = {}
        if action != -1:
            action_txt = []
            action_txt.append({"action":action, "start": 0, "end": counter})
            annotations_txt = {'actions':action_txt, 'frames': frames}
        else:
            annotations_txt = {'frames':frames}

        annotations = {'annotations':annotations_txt}
        with open(skeletonfile, 'w') as outfile:
            json.dump(annotations, outfile, indent = 4)

        cap.release()
