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
