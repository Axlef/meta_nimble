import cv2
import os
import argparse
import time
import imutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videofile', type = str, required = True)
    args = parser.parse_args()

    videofile = args.videofile
    if not os.path.isfile(videofile):
        raise IOError('invalid output file, .avi and/or .skeleton already exist')

    ### Webcam ###
    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        raise cv2.error('Unable to open video capture')
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = time.perf_counter()
    while(True):
        # Read new image and apply transformation
        ret, frame = cap.read()

        # Display frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1000//30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    end = time.perf_counter()
    print('time: {}'.format(end - start))
