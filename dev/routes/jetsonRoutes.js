var startStreamingf = require('../jetsonServer');
var stopCameraf = require('../jetsonServer');
var toggleSendImageFlagf = require('../jetsonServer');

/**
 * API communication
 * @param {*} app 
 */
module.exports = function (app) {
    app.post('/startStreaming', function(req,res){
   // console.log(req.body);
    startStreamingf.startStreaming(req,res);
    });
    app.post('/stopCamera', function(req,res){
    stopCameraf.stopCameraapi(req,res,function(err){});});
    app.post('/toggleSendImageFlag', function(req,res){toggleSendImageFlagf.toggleSendImageFlag(req,res);});
}
