import argparse
import logging
import time
import math
import cv2
import numpy as np
import click
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


from tf_pose.estimator import initialize_variables
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import requests
import json
import config
from sklearn import metrics
import random
from random import randint
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
url = config.serverPath
featureName = config.featureName
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

@click.command()
@click.argument('streamingurl')
@click.argument('camid')
@click.argument('devicename')

def main(streamingurl, camid, devicename):
    resize='0x0'
    resize_out_ratio=4.0
    model = 'mobilenet_thin'
    show_process=False
    fps_time = 0
    (sess, accuracy, pred, optimizer) = initialize_variables()
    #print(pred)
    
    #logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    if streamingurl.find("webcam") == 0:
        streamingurl = int(streamingurl[-1:])
    cam = cv2.VideoCapture(streamingurl)
    print("@@@@@@",streamingurl)
    ret_val, image = cam.read()
    frameRate = cam.get(5)
    print(frameRate)
    frameRate=math.floor(frameRate/8)
    #frameRate = 100
    
    frameId = 0
    frame_number=0
    sequence_arr=[]
    while True:
        ret_val, image = cam.read()
        if ret_val == True:
            if(frameId % frameRate ==0):
                #logger.debug('image process+')
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

                #logger.debug('postprocess+')
                sequence_arr,image, label = TfPoseEstimator.draw_humans(image, humans, frame_number,sequence_arr,imgcopy=False)
            
                try:
                    if label =='':
                        label="No Activity Detected"
                    #logger.debug("########", camid, devicename)
                    r = requests.post(url, data=json.dumps({"featureName":featureName,"label":label, "camId": camid, "deviceName": devicename}), headers=headers)
                except:
                    print("Check connection with the node server")
            
                frame_number= frame_number+1
                print("#",label)
            
                #logger.debug('show+')
                cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
                cv2.imshow('tf-pose-estimation result', image)
                fps_time = time.time()
                if cv2.waitKey(1) == 27:
                    break
                #logger.debug('finished+')
            frameId = frameId + 1
        else:
            cam = cv2.VideoCapture(streamingurl)

    cv2.destroyAllWindows()

if __name__ == '__main__':
     print("___________in wecam run__________")
     
     main()
