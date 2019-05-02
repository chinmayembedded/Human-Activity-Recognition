import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from tf_pose.estimator import initialize_variables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from sklearn import metrics
import random
from random import randint
import os


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    (sess, accuracy, pred, optimizer) = initialize_variables()
    print(pred)
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    #w, h = model_wh(args.resolution)
    #e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    cap = cv2.VideoCapture(args.video)
    frame_number=0
    sequence_arr=[]
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #humans = e.inference(image)
        #print("@@@@", humans)
        if not args.showBG:
            image = np.zeros(image.shape)
        sequence_arr,image, label = TfPoseEstimator.draw_humans(image, humans, frame_number,sequence_arr,imgcopy=False)
        frame_number= frame_number+1

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
