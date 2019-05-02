/**
 * For Jetson Node server code
 */
var config = {};

config.fpsNth = 10; //ex. if camera source fps if 30, algorithm will process the images every 30/10 = 3ms. i.e. 10fps
config.conf_thresh = 0.6; //helmet detection threshold - higher the value - better the accuracy

config.port = 5001;
config.name = "OpenPoseComputeEngine";
config.deviceType = "Nvidia Xavier";
config.cameraSupported = 3;
config.location = "GTC";
config.wayToCommunicate = "dev";
config.pingInterval = 360000;
config.IP = "IP of Jetson if multiple CE on one machine";
config.appendMac = "Compute";
config.imageQuality = 50;

/**
 * Backend configuration
 */
config.host = 'https://snsdevrestapp.azurewebsites.net/devices/computeengines';
config.IoTHubConnectionString = 'HostName=SnSDeviothub.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=vHrbGyhT+Vhq+iZ6uAtdywHObTxYPMEPBOIo0tZomVo=';


config.user = "/home/nvidia/";

//_________________________Configuration Done _____________________________________

config.detectionAlgorithms = [{
    "featureName": "textRecognition",
    "fps": 1,
    "shapeSupported": [1]
}];

//Backend URLs
config.sendDetectionResultUrl = "http://localhost:" + config.port + "/getresult";
config.JetsonRegistrationURL = config.host + "/devices/computeengines";
config.registerAlgorithm = config.host + "/devices/computeengines/algorithm";

//Folder paths
config.jetsondlFolderPath = '/home/nvidia/helmetDetection/';
config.poseDetectionFolder = '/home/nvidia/Pose-estimation/';
config.webappFolder  = '/home/nvidia/dist/assets/img/';
config.CamerasFolderPath = 'Cameras';
config.camFolder = config.webappFolder + config.CamerasFolderPath;
config.livestreamingCamFolder = config.camFolder + '/Cam';

config.DLmodel = config.jetsondlFolderPath + 'darknet/';
config.poseServerPath = config.poseDetectionFolder + '/run_webcam.py'

config.darknetDetectArg = 'detect';
config.darknetDetectorArg = 'detector';
config.darknetCfgArg = 'cfg/yolo.cfg';
config.darknetWeightFile = 'yolo.weights';
config.darknetThreshold = '0.35';
config.darknetHierThreshold = '0.5';
module.exports = config;
