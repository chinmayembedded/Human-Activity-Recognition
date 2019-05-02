var config = require('./settings');
var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var fs = require('fs');
var spawn = require('child_process').spawn;
var mkdirp = require('mkdirp');
var ip = require("ip");
var request = require("request");
var cors = require('cors');

var http = require('http').Server(app);
var io = require('socket.io')(http);

io.on('connection', function (socket) {
    console.log("Webapp client socket connected");
    socket.on('chat message', function (msg) {
        io.emit('chat message');
    });
});

io.on('disconnection', function (socket) {
    console.log("Webapp client socket connected");
    socket.on('chat message', function (msg) {
        io.emit('chat message');
    });
});

var cameraPID = [];
var blobCountMap = new Map();
var toggleSendImageMap = new Map();

//_________________________SERVER CONFIGURATION_________________________

app.use(bodyParser.urlencoded({
    extended: true
}));
app.use(bodyParser.json());
app.use(cors());

app.post('/_ping', function (req, res) {
    res.status(200).send(req.body);
});

http.listen(config.port, function () {
    console.log('Jetson Server is listening at: ' + config.port);
});

//___________________________Registration done_________________________
//Topic Names

//Compute Engine's ping Mechanism
console.log("Registering COMPUTE ENGINE on Server..................");
var macId = "MACIDXavier" + config.appendMac;
var jetsonDetails = {
    "name": config.name,
    "deviceType": config.deviceType,
    "macId": macId,
    "ipAddress": ip.address(),
    "detectionAlgoritms": config.detectionAlgoritms,
    "cameraSupported": config.cameraSupported,
    "location": config.location,
    "supportedShapes": config.supportedShapes,
    "wayToCommunicate": config.wayToCommunicate,
    "isCloudCompute": false
};

var options = {
    rejectUnauthorized: false,
    url: config.JetsonRegistrationURL,
    method: 'POST',
    json: jetsonDetails
};
request(options, function (error, response, body) {
    if (error) {
        console.log("Error Message : Error Registering the Compute Engine!");
    }
    else {
        var computeEngineId = response.body._id;

        /**
         * Algorithm Registration 
        */
        var algorithmDetails = {
            "computeEngineId": computeEngineId,
            "detectionAlgorithms": config.detectionAlgorithms,
        }

        var optionsAlgo = {
            rejectUnauthorized: false,
            url: config.registerAlgorithm,
            method: 'POST',
            json: algorithmDetails

        };

        request(optionsAlgo, function (error, response, body) {
            if (error) {
                console.log("Error Message : Error Registering the Compute Engine Algorithm!");
            } else {
                console.log("Success in Registering Algorithm!");
                pingMechanismInterval(jetsonDetails);
            }
        });
    }

});


var pingMechanismInterval = function (jetsonDetails) {

    setInterval(function () {
        var jsonToPing = jetsonDetails;

        var options = {
            rejectUnauthorized: false,
            url: config.JetsonRegistrationURL,
            method: 'POST',
            json: jsonToPing
        };
        request(options, function (error, response, body) {
            if (error) {
                console.log("\n**PING STATUS -> \n    Error in Ping Interval of the Compute Engine : ", error);
            } else {
                console.log("\n**PING STATUS -> \n    Success in Jetson Ping !");
            }
        });
    }, config.pingInterval);
}

//Handling Messages
/**
 * server backend and compute engine api communication
 */
require('./routes/jetsonRoutes')(app);

var startStreaming = function (req, res) {
    console.log("request body", req.body);
    res.end("start streaming call");
    console.log("start call");
    var sendData = req.body;
    var parsedJson = req.body;
    console.log(parsedJson);
    var camId = parsedJson.camId;
    var camArr = [];
    camArr.push(camId);
    stopCamera(camArr, function (msg) {
        console.log("Checking if same camera is already spawned! : ", msg);
        boundingBox(sendData, function (error) {
            if (!error) {
                console.log("IoTHub==================Create configuration files, and spawn DL model Done!!==========================\n");
                res.end("Start streaming call");
            }
            else
                console.log("Error Message : Error in Starting Process!");
        });
    });
}

var stopCameraapi = function (req, res) {
    res.end("Stop streaming call");
    console.log("Stop streaming call");
    var camIds = req.body.camIdArray;
    console.log("*STOP this cameras : ", camIds);
    stopCamera(camIds, function (error) {
        if (!error) {
            console.log("*Camera Stopped : ", error);
        }
        else {
            console.log("**Error in stopping cameras : ", error);
            //console.log("**ERROR :: ", error);
        }
    });
}

var toggleSendImageFlag = function (req, res) {
    res.end("Toggle streaming call");
    var toggleData = req.body;

    console.log(toggleData);
    if (toggleData.flag === 0 || toggleData.flag === 1) {
        toggleSendImageMap.set(toggleData.camId, toggleData.flag);
    } else {
        console.log("Error in ToggleSendImageFlag :: Invalid flag");
    }
    console.log(toggleSendImageMap);
}

/**
 * Creation of base directory for images
 */
if (!fs.existsSync(config.camFolder)) {
    mkdirp(config.camFolder, function (err) {
        if (err) {
            console.log('error in creating folder');
        }
        else {
            console.log("Base Directory created successfully!");
            //watchFunction();
        }
    });
}

//_________________________Functions_________________________
/**
 * Spawning the model 
 */
var spawnDLModel = function (camId, message, isLinePresent, isAnotherBox, isFace, feature, streamingUrl, deviceName) {
    console.log("_______________________________Compute Engine Darknet OUTPUT______________________________________");

    var darknet_args = config.darknetCfgArg + ' ' + config.darknetWeightFile + ' ' + config.darknetThreshold + ' ' + config.darknetHierThreshold;
    var DLmodel = config.DLmodel;
    var sendDetectionResultUrl = config.sendDetectionResultUrl;
    var livestreamingCamFolder = config.livestreamingCamFolder;
    var helmetServerPath = config.helmetServerPath;
    var darknet_arg_detect = config.darknetDetectArg;
    //message.boundingBox[0].detectionObjects = ["person"];
    //console.log("darknet args - \n", message);
    message = JSON.stringify(message);


    if (isFace) {
        toggleSendImageMap.set(camId, true);
        console.log("\n\n\nSpawning helmet faceRec server.....", message);
        const faceChild = spawn('python3', [helmetServerPath, message, livestreamingCamFolder,
            sendDetectionResultUrl, config.fpsNth, config.imageQuality, config.conf_thresh]
            //{ cwd: DLmodel }
        );

        //cameraId and PID storing
        cameraPID.push({ "cameraId": camId, "pid": faceChild.pid });

        faceChild.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
        });

        faceChild.stderr.on('data', (data) => {
            // console.log(`stderr: ${data}`);
        });

        faceChild.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });

    }
    else if(feature === "textRecognition"){
        const faceChild = spawn('python3', [config.poseServerPath, streamingUrl, camId, deviceName]); 
        cameraPID.push({ "cameraId": camId, "pid": faceChild.pid });
        console.log("Spawned Openpose model", camId);   
        faceChild.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
        });

        faceChild.stderr.on('data', (data) => {
             console.log(`stdout: ${data}`);
        });

        faceChild.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });    

    }
    else{
        if (isAnotherBox) {
            console.log("\n\n\nSpawning darknet.....", message);
            const child = spawn('./darknet',
                [darknet_arg_detect, darknet_args, sendDetectionResultUrl, livestreamingCamFolder, message],
                { cwd: DLmodel }
            );

            //cameraId and PID storing
            cameraPID.push({ "cameraId": camId, "pid": child.pid });

            child.stdout.on('data', (data) => {
                console.log(`stdout: ${data}`);
            });

            child.stderr.on('data', (data) => {
                console.log(`stderr: ${data}`);
            });

            child.on('close', (code) => {
                console.log(`child process exited with code ${code}`);
            });
        }

        if (isLinePresent) {
            const lineChild = spawn('./darknet',
                [config.darknetDetectorArg, darknet_args, sendDetectionResultUrl, livestreamingCamFolder, message],
                { cwd: DLmodel }
            );
            cameraPID.push({ "cameraId": camId, "pid": lineChild.pid });
            lineChild.stdout.on('data', (data) => {
                console.log(`stdout: ${data}`);
            });
            lineChild.on('close', (code) => {
                console.log(`line child process exited with code ${code}`);
            });
        }
    }
}

/**
 * to setup processing for camera
 * @param {*string} message 
 * @param {*function} callaback 
 */
var boundingBox = function (message, callback) {
    var parsedJson = (message);
    //console.log("@@@@@@@@@",message);
    var camId = parsedJson.camId;
    var deviceName = parsedJson.deviceName;
    var detection_type_str = parsedJson.feature;
    var bbox = (parsedJson.boundingBox);

    var isLinePresent = false;
    var isAnotherBox = false;
    var isFace = false;

    blobCountMap.set(camId, 1);

    bbox.forEach(function (box) {
        if (box.shape == 'Line')
            isLinePresent = true;
        else
            isAnotherBox = true;
    });

    if (parsedJson.feature == "faceRecognition")
        isFace = true;

    console.log("\n  CamId:::", camId);
    camera_folder = config.livestreamingCamFolder + camId;

    /*if (!fs.existsSync(camera_folder)) 
    {*/
    //create cameraId directory
    mkdirp(camera_folder, function (err) {
        if (err) {
            console.log("Error Message : Error in creating folder ", camera_folder);
        }
        else
            console.log(" Camera Directory created successfully!");
    });
    //}

    spawnDLModel(camId, message, isLinePresent, isAnotherBox, isFace, parsedJson.feature, parsedJson.streamingUrl,deviceName);
    callback(null);
}

/**
 * to stop processing cameras 
 * @param {*string} camIds 
 * @param {*function} callback 
 */
var stopCamera = function (camIds, callback) {
    var camId = camIds[0];
    var tempArr = cameraPID.slice();
    tempArr.forEach(function (cam, i) {

        if (cam.cameraId === camId) {
            toggleSendImageMap.delete(camId);
            console.log("Camera ID Found!");
            try {
                process.kill(cam.pid);
            } catch (e) {
                console.log("Process not found!");
            }
            console.log("The Process is Killed Succesfully!");
            cameraPID.splice(i, i + 1);
            callback(null);
        }
    });
    callback("notFound");
}

//________________DEMO code ________________
var fs = require('fs');

var poseEstimateJson = { 
     imageName: '5c5d69ba3965790390c78a47_textRecognition_1552370045599.jpg',
     bboxResults: [ { count: 1, markerName: 'text1', tagName: 'text2' } ],
     totalCount: 1,
     deviceName: '',
     timestamp: 1552370045599,
     feature: 'textRecognition',
     userId: 'snsuser@yopmail.com',
     camId: '',
     results: [],
     imageWidth: 1920,
     imageHeight: 1080,
     imageUrl: 'https://snsdevstrgaccnt.blob.core.windows.net/facethreeblobcontainer/5c5d69ba3965790390c78a47_textRecognition_1552370045599.jpg',
     //boundingBoxes: [ { line: '\\MN . 05. AS. 964', boundingBox: [], length: 17 } ],
     countPerBbox: [ { areaOfInterestId: undefined, count: 1 } ] 
}


app.post('/getResult', function (req, res) {
    res.end();
    //console.log("\nResult", poseEstimateJson.boundingBoxes[0].line);
    var resultData = req.body;
    if (req.body.featureName === "textRecognition"){
          //poseEstimateJson.boundingBoxes[0].line = req.body.label;
          poseEstimateJson["boundingBoxes"] = [{
                   'line' : req.body.label,
                    'boundingBox': [], 
                    'length': 20         
          }]
          poseEstimateJson["camId"] = req.body.camId;
          poseEstimateJson["deviceName"] = req.body.deviceName;
          resultData = poseEstimateJson;
    }
    //var resultData = JSON.stringify(req.body.imageName);

    //console.log(resultData);
    updateResult(resultData);
});

var countResults = 1;

var updateResult = function (resultData) {
    var imgArray = resultData.imageName.split('/');
    var imageNameLocal = imgArray[imgArray.length - 1];
    var imgFullPath = resultData.imageName;

    //var camId = imageNameLocal.split("_")[0];

    if (resultData.feature == 'humanDetection') {
        var liveDashboardData = {};
        liveDashboardData.camId = camId;
        liveDashboardData.deviceName = resultData.deviceName;
        liveDashboardData.count = resultData.totalCount;
        liveDashboardData.timestamp = new Date().getTime();
        liveDashboardData.bboxResults = resultData.bboxResults;
        /**Send Live Data to dashboard*/
        io.emit('liveDashboard/' + resultData.userId, {
            message: liveDashboardData
        });
    }

    resultData.imageName = imageNameLocal;
    resultData.imgBase64 = imgFullPath.replace("/home/nvidia/dist", "");
    resultData.bbox = resultData.boundingBoxes;
    delete resultData["boundingBoxes"];
    resultData.totalResult = resultData.totalCount;
    delete resultData["totalCount"];
    resultData.countPerBox = resultData.countPerBbox;
    delete resultData["countPerBbox"];
    //resultData.camId = camId;

    //console.log(resultData);

    /**Send Live Data */
    io.emit('liveImage', {
        message: resultData
    });
    if (countResults == 30) {
        //console.log("Result -", resultData);
        countResults = 1;
    }
    else
        countResults = countResults + 1
}

module.exports.toggleSendImageFlag = toggleSendImageFlag;
module.exports.stopCameraapi = stopCameraapi;
module.exports.startStreaming = startStreaming;
