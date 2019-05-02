/**
 * For Jetson Node server code
 */
var config = {};

config.fpsNth = 15; //ex. if camera source fps if 30, algorithm will process the images every 30/10 = 3ms. i.e. 10fps
config.conf_thresh = 0.6; //helmet detection threshold - higher the value - better the accuracy

config.port = 5001;
config.name = "TestingCompute";
config.deviceType = "Nvidia Xavier";
config.cameraSupported = 3;
config.location = "<Location>";
config.wayToCommunicate = "standalone";
config.pingInterval = 360000;
config.IP = "IP of Jetson if multiple CE on one machine";
config.appendMac = "Compute";
config.imageQuality = 50;

/**
 * Backend configuration
 */
config.host = 'https://localhost:2500';

config.user = "/home/nvidia/";

//_________________________Configuration Done _____________________________________

config.detectionAlgorithms = [{
    "featureName": "humanDetection",
    "fps": 1,
    "shapeSupported": [1]
}, {
    "featureName": "faceRecognition",
    "fps": 1,
    "shapeSupported": [1]
}, {
    "featureName": "objectDetection",
    "fps": 1,
    "shapeSupported": [1],
    "objectSupported": ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]
}];

//Backend URLs
config.sendDetectionResultUrl = "http://localhost:" + config.port + "/getresult";
config.JetsonRegistrationURL = config.host + "/devices/computeengines";
config.registerAlgorithm = config.host + "/devices/computeengines/algorithm";

//Folder paths
config.jetsondlFolderPath = '/home/nvidia/helmetDetection/';
config.helmetDetectionFolder = '/home/nvidia/helmetDetection/';
config.poseDetectionFolder = '/home/nvidia/Pose-estimation/';
config.webappFolder  = '/home/nvidia/dist/assets/img/';
config.CamerasFolderPath = 'Cameras';
config.camFolder = config.webappFolder + config.CamerasFolderPath;
config.livestreamingCamFolder = config.camFolder + '/Cam';

config.DLmodel = config.jetsondlFolderPath + 'darknet/';
config.helmetServerPath = config.helmetDetectionFolder + '/server.py';
config.poseServerPath = config.poseDetectionFolder + '/run_webcam.py';

config.darknetDetectArg = 'detect';
config.darknetDetectorArg = 'detector';
config.darknetCfgArg = 'cfg/yolo.cfg';
config.darknetWeightFile = 'yolo.weights';
config.darknetThreshold = '0.35';
config.darknetHierThreshold = '0.5';
module.exports = config;






