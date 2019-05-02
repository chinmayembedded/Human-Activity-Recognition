var config = require('./settings');
var spawn = require('child_process').spawn;

var request = require('request');
var iothub = require('azure-iothub');
var fs = require('fs');
var http = require('http');

const path = require('path');
var clientFromConnectionStringAMQP = require('azure-iot-device-amqp').clientFromConnectionString;
var ip = require("ip");
var serial = require('node-serial-key');
var mkdirp = require('mkdirp');
var cameraPID = new Map();


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
    
    //iotHubClient.on('message', function (message) {

        // console.log('Id: ' + message.messageId + ' Body: ' + message.data);

        //iotHubClient.complete(message, printResultFor('completed'));


      /*  var topic = message.messageId;
        var message = message.data;

        var messageJSON = message.toString();
        var parsedMessage = JSON.parse(messageJSON);


        console.log('\n Topic: \n');
        console.log("-----------------_>",topic);

        console.log('\n Message: \n')
        //console.log(parsedMessage);

        //console.log('\n Message Body: \n');
        //console.log(parsedMessage.body);
        //console.log('\n');
*/
var topic = "startStreaming";
        switch (topic) {
            case '/':
                {
                    console.log("MQTT==================Project Heimdall Server Available to Respond!!\n-----------------------------------\n");
                    break;
                }

            case "startStreaming":
                {
                    console.log("Starting Streaming..................");


                    const faceChild = spawn('python3', [config.poseServerPath]);    

                    /* To stop camera add it's PID in the map*/
                    //cameraPID.set(parsedMessage.camId, poseChild.pid);

        faceChild.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
        });

        faceChild.stderr.on('data', (data) => {
            // console.log(`stderr: ${data}`);
        });

        faceChild.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });


			    //If we start pumping the data in real time, UI hangs.
			/*if ((message_count%5)!=0)
			{                            
                                return;
                        }
                        
			var data = data + '';

                        if (data.indexOf("boxes")) {
                            
                            // we have found a match

                            var arr = data.split(" ");
                            
                            // console.log(arr);
                            // One ugly assumption here based on the prints we have added. If changing the command line, make sure you change the array param
                            
                            var output =  arr[3];
		            var output_source = arr[5];
                            var num_source = parseInt(output_source);
                            //console.log(" ", num_source);
			    switch(num_source){
				case 0:
					//console.log("0 : "+parseInt(output_source));
					//desired_output.imageName = parsedMessage_1.camId;
					//parsedMessage.camId = parsedMessage_1.camId;
					parsedMessage = parsedMessage_1;
					//console.log("0 : "+parsedMessage.deviceName);
					break;
				case 1: 

					//parsedMessage.camId = parsedMessage_2.camId;
					//desired_output.imageName = parsedMessage_2.camId;
					parsedMessage = parsedMessage_2;
					//console.log("           1 : "+parsedMessage.deviceName);
					break;
				case 2: 
					//console.log("                           2 : "+parseInt(output_source));
					//parsedMessage.camId = parsedMessage_3.camId;
					//desired_output.imageName = parsedMessage_3.camId;
					parsedMessage = parsedMessage_3;
					//console.log("                           2 : "+parsedMessage.deviceName);
					break;
				case 3: 
					//console.log("                                           3 : "+parseInt(output_source));
					//parsedMessage.camId = parsedMessage_4.camId;
					//desired_output.imageName = parsedMessage_4.camId;
					parsedMessage = parsedMessage_4;
					//console.log("                                           3 : "+parsedMessage.deviceName);
					break;
				default:
					console.log("Default: I got output_source as: "+parseInt(output_source));

				}
			        //console.log("I got output_source as: "+parseInt(output_source));

	// AB: end hardcoding

                            if (isNumber(output)) {
                                if(parsedMessage.boundingBox){
                                var desired_output = {

                                    bboxResults: [
                                        {
                                        count: output,
                                        markerName: parsedMessage.boundingBox[0].markerName,
                                        tagName: parsedMessage.boundingBox[0].tagName
                                        }
                                    ],
                                    totalCount: output,
                                    deviceName: parsedMessage.deviceName,
                                    feature: parsedMessage.feature,
                                    userId: parsedMessage.userId,
                                    imageName: parsedMessage.camId+"_"+new Date().getTime() + ".jpg",
                                    boundingBoxes: [
                                    {
                                        bboxes: {
                                        x1: parsedMessage.boundingBox[0].x ,
                                        y1: parsedMessage.boundingBox[0].y ,
                                        x2: parsedMessage.boundingBox[0].x2,
                                        y2: parsedMessage.boundingBox[0].y2
                                        },
                                        objectType: 'person',
                                        tagName: parsedMessage.boundingBox[0].tagName,
                                        markerName: parsedMessage.boundingBox[0].markerName
                                    },
                                    {
                                        objectType: 'person',
                                        bboxes: {
                                        x1: parsedMessage.boundingBox[0].x,
                                        y1: parsedMessage.boundingBox[0].y,
                                        x2: parsedMessage.boundingBox[0].x2,
                                        y2: parsedMessage.boundingBox[0].y2
                                        },
                                        tagName: parsedMessage.boundingBox[0].tagName,
                                        markerName: parsedMessage.boundingBox[0].markerName
                                    },
                                    {
                                        objectType: 'person',
                                        bboxes: {
                                        x1: parsedMessage.boundingBox[0].x,
                                        y1: parsedMessage.boundingBox[0].y,
                                        x2: parsedMessage.boundingBox[0].x2,
                                        y2: parsedMessage.boundingBox[0].y2
                                        },
                                        tagName: parsedMessage.boundingBox[0].tagName,
                                        markerName: parsedMessage.boundingBox[0].markerName
                                    }
                                    ],
//                                    imageUrl: 'https://snsdevstrgaccnt.blob.core.windows.net/facethreeblobcontainer/5c5d687b3965790390c789d7.jpg'
   				      imageUrl: ''
				}

//				desired_output.totalCount = '13';

				//console.log(desired_output);
                                
                                var message = new Message (JSON.stringify(desired_output));

                                message.ack = 'full';
                                //console.log('\n+++++++ Sending desired output to cloud');
                                
                              iotHubClient.sendEvent(message, function (err) {
                                    if (err) {
                                      console.error('Could not send: ' + err.toString());
                                     
                                      //process.exit(-1);

                                    } else {
                                      //console.log('Message sent: ' + JSON.stringify(desired_output));
                                      
                                      //process.exit(0);
                                    
                                    }
                                });
                                }
                            }
                    
                        }
                    })
                }*/
                break;
            /*case "stopCamera":
                {
                        var camIds = json.PARSE(message.toString());
                        console.log("\n*Stop these cameras ::", typeof(camIds));
                        stopCamera(camIds, function(err){
                                console.log("Stopped - > ", camIds);                        
                        });
                }
                break;*/
        }}
   // })
};

/**
 * to stop processing cameras 
 * @param {*string} camIds 
 * @param {*function} callback 
 */
var stopCamera = function (camIds, callback) {
    var camId = camIds[2];
    console.log("Camera ------------->", camId);
    try {
        process.kill(cameraPID.get(camId));
    } catch (e) {
        console.log("Process not found!");
    }
    callback("notFound");
}

                           


                    

       
