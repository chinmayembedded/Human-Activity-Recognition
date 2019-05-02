var config = require('./settings');
var spawn = require('child_process').spawn;
var express = require('express');
var request = require('request');
var iothub = require('azure-iothub');
var fs = require('fs');
var http = require('http');
var app = express();
var bodyParser = require('body-parser');
const path = require('path');
var clientFromConnectionStringAMQP = require('azure-iot-device-amqp').clientFromConnectionString;
var ip = require("ip");
var serial = require('node-serial-key');
var mkdirp = require('mkdirp');

var cors = require('cors');

var http = require('http').Server(app);

http.listen(config.port, function () {
    console.log('Jetson Server is listening at: ' + config.port);
});

var cameraPID = new Map();
app.use(bodyParser.urlencoded({
    extended: true
}));
app.use(bodyParser.json());
app.use(cors());

// Globals
var connectionString = config.IoTHubConnectionString;
var deviceConnectionString = 0;
var registry = iothub.Registry.fromConnectionString(connectionString);
var Client = require('azure-iot-device').Client;
var Protocol = require('azure-iot-device-mqtt').Mqtt;
var Message = require('azure-iot-device').Message;


var iotHubClient = 0;

var has_compute_started=0;

var message_count = 0
var parsedMessage_1=0, parsedMessage_2=0, parsedMessage_3=0, parsedMessage_4=0;


serial.getSerial(function (err, value) {

    console.log('\n This is value from serial.getSerial Function, which is the macID: ' +  value + '\n');
    console.log("Registering COMPUTE ENGINE on Server..................");

    var macId = value + "pose";
    var movidiusDetails = {
        "name": config.name,
        "deviceType": "NVIDIA Xavier with OpenPose",
        "macId": macId,
        "ipAddress": ip.address(),
        "detectionAlgorithms": config.detectionAlgorithms,
        "cameraSupported": 3,
        "location": config.GTC,
        "supportedShapes": 0,
        "wayToCommunicate": "rsync"
    };

    var options = {
        rejectUnauthorized: false,
        url: config.host,
        method: 'POST',
        json: movidiusDetails
    };

    request(options, function (error, response, body) {
        if (error) {
            console.log("Error Message : Error Registering the Compute Engine!");
        }
        else {

            var computeEngineId = response.body._id;

            // Create a new device
            var device = {
                deviceId: computeEngineId
            };

            registry.create(device, function (err, deviceInfo, res) {
                if (err) {
                    console.log(err.toString());
                    registry.get(device.deviceId, function (err, deviceInfo, res) {
                        console.log("Device Already Registered");
                        deviceConnectionString = connectionString.split(';')[0] + ";DeviceId=" + deviceInfo.deviceId + ";SharedAccessKey=" + deviceInfo.authentication.symmetricKey.primaryKey;
                        console.log(deviceConnectionString);
                        iotHubConnection(deviceConnectionString);
                    });
                }

                if (res) console.log(' status: ' + res.statusCode + ' ' + res.statusMessage);
                if (deviceInfo) {
                    deviceConnectionString = connectionString.split(';')[0] + ";DeviceId=" + deviceInfo.deviceId + ";SharedAccessKey=" + deviceInfo.authentication.symmetricKey.primaryKey;
                    console.log(deviceConnectionString);
                    iotHubConnection(deviceConnectionString);
                }
            });

            /**
             * Algorithm Registration 
            */
            var algorithmDetails = config.detectionAlgorithms;

            var optionsAlgo = {
                rejectUnauthorized: false,
                url: config.host,
                method: 'POST',
                json: algorithmDetails
            };

            request(optionsAlgo, function (error, response, body) {
                if (error) {
                    console.log("Error Message : Error Registering the Compute Engine Algorithm!");
                } else {
                    console.log("Success in Registering Algorithm!");
                    pingMechanismInterval(movidiusDetails);
                }
            });
        }
    });
});



var pingMechanismInterval = function (movidiusDetails) {

    setInterval(function () {
        var jsonToPing = movidiusDetails;

        var options = {
            rejectUnauthorized: false,
            url: config.host,
            method: 'POST',
            json: jsonToPing
        };
        request(options, function (error, response, body) {
            if (error) {
                console.log("\n**PING STATUS -> \n    Error in Ping Interval of the Compute Engine : ", error);
            } else {
                console.log("\n**PING STATUS -> \n    Success in Movidius Ping !");
            }
        });
    }, 360000);
}

var iotHubConnection = function (deviceConnectionString) {
    console.log("Connecting to IoTHub...");

     iotHubClient = Client.fromConnectionString(deviceConnectionString, Protocol);
     iotHubClient.open(IOTHubListener);

    //To reconnect to IOTHub to solve disconnect issue
    setInterval(function () {
        iotHubClient.removeAllListeners();
        iotHubClient.close();
        delete iotHubClient;
        iotHubClient = Client.fromConnectionString(deviceConnectionString, Protocol);
        iotHubClient.open(IOTHubListener);
    }, 900000);  //15minutes
}

function printResultFor(op) {
    return function printResult(err, res) {
        if (err) console.log(op + ' error: ' + err.toString());
        if (res) console.log(op + ' status: ' + res.constructor.name);
    };
}

//Function isNumber - Used in IOTHubListener in 'Start Streaming' Case

function isNumber(n) { return !isNaN(parseFloat(n)) && !isNaN(n - 0) }

//IoTHubListener Function 

var IOTHubListener = function () {

    console.log("Connected to IoTHub.")
    
    iotHubClient.on('message', function (message) {

        // console.log('Id: ' + message.messageId + ' Body: ' + message.data);

        iotHubClient.complete(message, printResultFor('completed'));
        var topic = message.messageId;
        var message = message.data;

        var messageJSON = message.toString();
        var parsedMessage = JSON.parse(messageJSON);


        console.log('\n Topic: \n');
        console.log("-----------------_>",topic);

        console.log('\n Message: \n', parsedMessage)
        //console.log(parsedMessage);
        switch (topic) {
               case '/':
                {
                    console.log("MQTT==================Project Heimdall Server Available to Respond!");
                    break;
                }
            case "startStreaming":
                {
                    console.log("Starting Streaming..................");

                    console.log(message)
                    const poseChild = spawn('python3', [config.poseServerPath, parsedMessage.streamingUrl, parsedMessage.camId, parsedMessage.deviceName]);    

                    /* To stop camera add it's PID in the map*/
                    //cameraPID.push({ "cameraId": camId, "pid": poseChild.pid });
                         cameraPID.set(parsedMessage.camId, poseChild.pid);
                    poseChild.stdout.on('data', function(data) {
                         console.log(data);
                    })
                    poseChild.stderr.on('data', (data) => {
                         console.log(`stdout: ${data}`);
                    });
               }
                break;
            case "stopCamera":
                {
                        var camIds = JSON.parse(message.toString());
                        console.log("\n*Stop these cameras ::", typeof(camIds));
                        stopCamera(camIds, function(err){
                                console.log("Stopped - > ", camIds);                        
                        });
                }
                break;
        }
    })
};

/**
 * to stop processing cameras 
 * @param {*string} camIds 
 * @param {*function} callback 
 */
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
     imageUrl: '',
     //boundingBoxes: [ { line: '\\MN . 05. AS. 964', boundingBox: [], length: 17 } ],
     countPerBbox: [ { areaOfInterestId: undefined, count: 1 } ] 
}

app.post('/getResult', function (req, res) {
    res.end();
    console.log("\nResult",req.body);
    
    var resultData = req.body;
     //console.log("\nResult",resultData);

    if (req.body.featureName === "textRecognition"){
          //poseEstimateJson.boundingBoxes[0].line = req.body.label;
          poseEstimateJson["boundingBoxes"] = [{
                   'line' : req.body.label,
                    'boundingBox': [], 
                    'length': 20         
          }]
          poseEstimateJson["deviceName"]= req.body.deviceName;
          poseEstimateJson["camId"]= req.body.camId;
          poseEstimateJson["imageName"] = req.body.camId + "_textRecognition_1552370045599.jpg"
          resultData = poseEstimateJson;
    }
    //var resultData = JSON.stringify(req.body.imageName);

    //console.log(resultData);
    sendResult(resultData);
});

var sendResult = function (resultData) {
   
   var message = new Message (JSON.stringify(resultData));
     console.log("!!!!!!!!!!Result Data", JSON.stringify(resultData));
   message.ack = 'full';
                            
                                
   iotHubClient.sendEvent(message, function (err) {
      if (err) {
           console.error('Could not send: ' + err.toString());
                                     
           //process.exit(-1);

      } else {
          console.log("Data sent::")
                                    
      }
   });
}

var stopCamera = function (camIds, callback) {
        var camId = camIds[0];
    console.log("Camera ------------->", camIds);
    try {
        process.kill(cameraPID.get(camId));
        //has_compute_started = 0;
        callback("Camera stopped");
    } catch (e) {
        console.log("Process not found!");
        callback("notFound");
    }

}

                           


                    

       
