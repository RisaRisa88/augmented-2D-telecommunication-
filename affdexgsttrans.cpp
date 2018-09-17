// Compile with: $ g++ opencv_gst.cpp -o opencv_gst `pkg-config --cflags --libs opencv`
//compile with : $g++ affdexgsttrans.cpp -o ./affdexgsttrans -std=c++11 -I$HOME/affdex-sdk/include -L$HOME/affdex-sdk/lib -lopencv_core `pkg-config --cflags --libs opencv` -I/usr/local/include -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/ -lboost_program_options -ltensorflow -laffdex-native

#include "CameraDetector.h"
#include "Face.h"
#include "Frame.h"
#include "PhotoDetector.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <string>
#include <atomic>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <stdlib.h>

//#include "MJPEGWriter.h"

using namespace boost::program_options;
using namespace std;
using namespace cv;
using namespace affdex;


Mat filter;
Mat filter1=imread("/home/takumi/affdex-sdk/data/screentones/2happy.png",1);
Mat filter2=imread("/home/takumi/affdex-sdk/data/screentones/1anger.png",1);
Mat filter3=imread("/home/takumi/affdex-sdk/data/screentones/3depression.png",1);
Mat filter4=imread("/home/takumi/affdex-sdk/data/screentones/4shock.png",1);

//Callback class for Face(Found or Lost)
class FListener: public affdex::FaceListener {

  void onFaceFound(float timestamp, affdex::FaceId faceId) {

  }
  void onFaceLost(float timestamp, affdex::FaceId faceId) {

  }

};





// IListenerを定義すること

//Callback class for image results
class IListener: public affdex::ImageListener {
  /// OnImageResultsの中で...
  //// IListnerでは、affdexの処理の結果を元に画像を加工する
  //// writerに書き込んで、外におくる

  std::mutex mMutex;
public:
  cv::VideoWriter writer;
 
  /*
  IListener(){
	writer.open("appsrc ! videoconvert ! x264enc noise-reduction=10000 tune=zerolatency byte-stream=true threads=4 ! mpegtsmux ! udpsink host=localhost port=9999", 0, (double)30, cv::Size(640, 480), true);
	if (!writer.isOpened()) {
  	printf("=ERR= can't create video writer\n");
	}
  }
  */
 
  void onImageResults(std::map<affdex::FaceId, affdex::Face> faces, affdex::Frame image){
	std::lock_guard<std::mutex> lg(mMutex);

	//cv::Mat videoFrame = *image.getImage();

	// 感情の結果に合わせて，フィルタを画像に付与する
	// 	↓
	// 画像を送る

	for(auto pair : faces){
  	auto face = pair.second;
  	float anger = face.emotions.anger;
  	float joy = face.emotions.joy;
  	float sadness = face.emotions.sadness;
  	float surprise = face.emotions.surprise;


  	cout<<"joy: " <<joy<< "\n anger: "<<to_string(anger).c_str()<<  "\n sadness: "<< to_string(sadness).c_str()<< "\n surprise: " << to_string(surprise).c_str()<<std::endl;

  	if(joy>anger && joy>sadness && joy>surprise){
    filter=filter1;
  	}else if (anger>joy && anger>sadness && anger>surprise){
    filter=filter2;
  	}else if (surprise>joy && surprise>sadness && surprise>joy){
    filter=filter4;
  	}else if (sadness>joy && sadness>anger && sadness>surprise){
    filter=filter3;
  	}
 	 
  	//filter.copyTo(videoFrame(cv::Rect(0,0,1024,640)));
  	//writer << videoFrame;
	}
  }

  void onImageCapture(affdex::Frame image){
	std::lock_guard<std::mutex> lg(mMutex);
  };
};

 void DrawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt)
{
    cv::Mat img_rgb, img_aaa, img_1ma;
    vector<cv::Mat>planes_rgba, planes_rgb, planes_aaa, planes_1ma;
    int maxVal = pow(2, 8*baseImg.elemSize1())-1;
 
    //透過画像はRGBA, 背景画像はRGBのみ許容。ビット深度が同じ画像のみ許容
    if(transImg.data==NULL || baseImg.data==NULL || transImg.channels()<4 ||baseImg.channels()<3 || (transImg.elemSize1()!=baseImg.elemSize1()) )
    {
        img_dst = cv::Mat(100,100, CV_8UC3);
        img_dst = cv::Scalar::all(maxVal);
        return;
    }
 
    //書き出し先座標が指定されていない場合は背景画像の中央に配置する
    if(tgtPt.size()<4)
    {
        //座標指定(背景画像の中心に表示する）
        int ltx = (baseImg.cols - transImg.cols)/2;
        int lty = (baseImg.rows - transImg.rows)/2;
        int ww  = transImg.cols;
        int hh  = transImg.rows;
 
        tgtPt.push_back(cv::Point2f(ltx   , lty));
        tgtPt.push_back(cv::Point2f(ltx+ww, lty));
        tgtPt.push_back(cv::Point2f(ltx+ww, lty+hh));
        tgtPt.push_back(cv::Point2f(ltx   , lty+hh));
    }
 
    //変形行列を作成
    vector<cv::Point2f>srcPt;
    srcPt.push_back( cv::Point2f(0, 0) );
    srcPt.push_back( cv::Point2f(transImg.cols-1, 0) );
    srcPt.push_back( cv::Point2f(transImg.cols-1, transImg.rows-1) );
    srcPt.push_back( cv::Point2f(0, transImg.rows-1) );
    cv::Mat mat = cv::getPerspectiveTransform(srcPt, tgtPt);
 
    //出力画像と同じ幅・高さのアルファ付き画像を作成
    cv::Mat alpha0(baseImg.rows, baseImg.cols, transImg.type() );
    alpha0 = cv::Scalar::all(0);
    cv::warpPerspective(transImg, alpha0, mat,alpha0.size(), cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
 
    //チャンネルに分解
    cv::split(alpha0, planes_rgba);
 
    //RGBA画像をRGBに変換   
    planes_rgb.push_back(planes_rgba[0]);
    planes_rgb.push_back(planes_rgba[1]);
    planes_rgb.push_back(planes_rgba[2]);
    merge(planes_rgb, img_rgb);
 
    //RGBA画像からアルファチャンネル抽出   
    planes_aaa.push_back(planes_rgba[3]);
    planes_aaa.push_back(planes_rgba[3]);
    planes_aaa.push_back(planes_rgba[3]);
    merge(planes_aaa, img_aaa);
 
    //背景用アルファチャンネル   
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    merge(planes_1ma, img_1ma);
 
    img_dst = img_rgb.mul(img_aaa, 1.0/(double)maxVal) + baseImg.mul(img_1ma, 1.0/(double)maxVal);
}


int main(int argc, char** argv) {

	// Original gstreamer pipeline:
	//  	== Sender ==
	//  	gst-launch-1.0 v4l2src
	//  	! video/x-raw, framerate=30/1, width=640, height=480, format=RGB
	//  	! videoconvert
	//  	! x264enc noise-reduction=10000 tune=zerolatency byte-stream=true threads=4
	//  	! mpegtsmux
	//  	! udpsink host=localhost port=5000
	//
	//  	== Receiver ==
	//  	gst-launch-1.0 -ve udpsrc port=5000
	//  	! tsparse ! tsdemux
	//  	! h264parse ! avdec_h264
	//  	! videoconvert
	//  	! ximagesink sync=false

	// first part of sender pipeline
	//cv::VideoCapture cap("v4l2src ! video/x-raw, framerate=30/1, width=640, height=480, format=RGB ! videoconvert ! appsink");
cv::VideoCapture cap("v4l2src ! videoconvert ! appsink");    
	if(!cap.isOpened()) {
    	printf("=ERR= can't create video capture\n");
    	return -1;
	}

cv::VideoWriter writer;
    writer.open("appsrc ! videoconvert ! autovideosink", 0, (double)30, cv::Size(640, 480), true);
    if (!writer.isOpened()) {
        printf("=ERR= can't create video writer\n");
        return -1;
    }

	cv::Mat transImg, baseImg, img_dst;
    cv::Mat frame;
    int key;

	//画像読み込み
    transImg   = cv::imread("/home/takumi/affdex-sdk/data/screentones/4shock.png", cv::IMREAD_UNCHANGED);
    //baseImg = cv::imread("/home/takumi/affdex-sdk/happiness.jpg");
 
    if( (transImg.data==NULL) )
    {
        printf("------------------------------\n");
        printf("image not exist\n");
        printf("------------------------------\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("------------------------------\n");
        printf("Press ANY key to quit\n");
        printf("------------------------------\n");
    }

	//座標指定(背景画像の中心に表示する）
    int ltx = (baseImg.cols - transImg.cols)/2;
    int lty = (baseImg.rows - transImg.rows)/2;
    int ww  = transImg.cols;
    int hh  = transImg.rows;
    vector<cv::Point2f>tgtPt;

	// affdexのphotoDetectorを初期化してセット

    	/////////////////////
    	// Affdex Settings //
    	/////////////////////
    

    	affdex::path DATA_FOLDER = "/home/takumi/affdex-sdk/data";

    	//Initialize camera detector
    affdex::PhotoDetector detector(2,affdex::FaceDetectorMode::LARGE_FACES);
    //std::shared_ptr<Detector> detector;
    //detector = std::make_shared<PhotoDetector>(nFaces,(affdex::FaceDetectorMode) FaceDetectorMode);

    	//Set locations of emotion models
    	detector.setClassifierPath(DATA_FOLDER);

    	//Enable classifiers to track
    	detector.setDetectAllEmotions(true);
    	detector.setDetectAllEmojis(false);
    	detector.setDetectAllExpressions(false);

    	//Add callbacks to detector
    	std::shared_ptr<affdex::FaceListener> fListen(new FListener());
    	//std::shared_ptr<affdex::ImageListener> iListen(new IListener());
    	std::shared_ptr<affdex::ImageListener> iListen(new IListener());

    //((IListener *)iListen.get())->writer.open("appsrc ! videoconvert ! x264enc noise-reduction=10000 tune=zerolatency byte-stream=true threads=4 ! mpegtsmux ! autovideosink", 0, (double)30, cv::Size(640, 480), true);

//((IListener *)iListen.get())->writer.open("appsrc ! videoconvert ! autovideosink", 0, (double)30, cv::Size(640, 480), true);
    
    	detector.setFaceListener(fListen.get());
    	detector.setImageListener(iListen.get());

  detector.start();

    // VideoCapture cap;
    	bool ok = cap.open(0);
    	if (!ok)
    	{
        	printf("no cam found ;(.\n");
        	pthread_exit(NULL);
    	}


	cv::Mat img;
	//cv::Mat frame;
	//int key;

	while (true) {

		cap >> baseImg;
		if(baseImg.empty()) break;
 
    DrawTransPinP(img_dst, transImg, baseImg, tgtPt)  ;

        writer << img_dst;
        key = cv::waitKey( 30 );
	}

}




