/* compile command
g++ -Wall newmain0-3.cpp -o newmain0-3 $(pkg-config --cflags --libs gstreamer-1.0) $(pkg-config --libs --cflags gstreamer-app-1.0) $(pkg-config --cflags --libs opencv) -std=c++11 -I$HOME/affdex-sdk/include -L$HOME/affdex-sdk/lib -lopencv_core -I/usr/local/include -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/ -lboost_program_options -ltensorflow -laffdex-native -Wno-unknown-pragmas -O2
*/
#include <stdio.h>
#include <stdlib.h>
#include <gst/gst.h>
#include <string.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstpipeline.h>
#include <gst/gstutils.h>
//for affdex
#include "CameraDetector.h"
#include "Face.h"
#include "Frame.h"
//#include "FrameDetector.h"
#include "PhotoDetector.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <string>
#include <atomic>
#include <mutex>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>

#include <unordered_map>

//check time delays
#include <sys/time.h>


struct timeval start_time, now, _time;

using namespace cv;
using namespace boost::program_options;
using namespace std;
using namespace affdex;

const gchar* p_src =
  "v4l2src v4l2=0  do-timestamp=1 is-live=1 name=mux ! "
  // "videotestsrc is-live=1 ! "
  "video/x-raw,width=320,height=240,format=BGR,framerate=15/1 !"
  //"identity name=point1 !"
  "appsink name=testsink";
  
const gchar* p_sink =
  "appsrc name=testsource caps=video/x-raw,width=320,height=240,format=BGR,framerate=15/1 !"
  //"identity name=point2 !"
  /*
  "videoconvert ! "
  "identity name=point3 !"
  "ximagesink";
  */
  " queue ! videoconvert ! queue !"
  " vp8enc error-resilient=1 speed-present=ultrafast max-latency=2 quality=2 speed=7 cpu-used=10 end-usage=cbr target-bitrate=1024000 token-partitions=3 static-threshold=1000 min-quantizer=0 max-quantizer=63 threads=2 deadline=50 keyframe-max-distance=20 ! queue ! rtpvp8pay pt=96 ! queue !"
  " identity name=point3 ! "
  " udpsink host=127.0.0.1 port=5004 auto-multicast=true "
  " autoaudiosrc buffer-time=36000  name=src src. ! audioconvert ! audioresample ! audio/x-raw,rate=16000,channels=1, format=S16LE ! "
  " opusenc bitrate=20000 !"
  " queue ! rtpopuspay ! queue ! "
  " udpsink host=127.0.0.1 port=5002 ";
  //" udpsink host=127.0.0.1 port=5002";
ofstream myfile;

typedef struct
{
  GMainLoop *loop;
  GstElement *source;
  GstElement *sink;
  //affdex::FrameDetector *detector;
  affdex::PhotoDetector *detector;
  //struct timeval *_time;
} ProgramData;

struct timeval e_start_time, e_now;

cv::Mat img_rgb[6], img_aaa[6], img_1ma[6];

void point1_handoff_handler (GstElement* object, GstBuffer* arg0, gpointer *data)
{
  GstMapInfo info;

  static int i=1;
  (void)object;
  (void)data;
  
  gst_buffer_map (arg0, &info, GST_MAP_READ);
  
  g_print ("point1: %02d\n", i++);
  g_print ("  pts : %" G_GUINT64_FORMAT "\n", arg0->pts);
  g_print ("  mono: %" G_GINT64_FORMAT "\n", g_get_monotonic_time());
  
}

void point2_handoff_handler (GstElement* object, GstBuffer* arg0, ProgramData *data)
{
  GstMapInfo info;
  
  static int i=1;
  (void)object;
  (void)data;
  
  gst_buffer_map (arg0, &info, GST_MAP_READ);
  
  g_print ("point2: %02d\n", i++);
  g_print ("  pts : %" G_GUINT64_FORMAT "\n", arg0->pts);
  g_print ("  mono: %" G_GINT64_FORMAT "\n", g_get_monotonic_time());

}

void point3_handoff_handler (GstElement* object, GstBuffer* arg0, ProgramData *data)
{
  double sec  = (double)_time.tv_sec;
  double micro = (double)_time.tv_usec;

  gettimeofday(&_time ,NULL);
  
  sec = (double)_time.tv_sec - sec;
  micro = (double)_time.tv_usec - micro;
  
  double passed = 1/(sec + micro / 1000.0 / 1000.0);
  cout<<"FPS : "<<to_string(passed).c_str()<<std::endl;

}


Mat filter, filtera, filter2, filter3, filter4, filter5, filter6, nofilter;

//Callback class for Face(Found or Lost)
class FListener: public affdex::FaceListener {
  void onFaceFound(float timestamp, affdex::FaceId faceId) {
  }
  void onFaceLost(float timestamp, affdex::FaceId faceId) {
  }
};
//Callback class for image results
class IListener: public affdex::ImageListener {
  //std::mutex mMutex;

  int filter_type = 0;
  std::mutex m_ft;

  void set_ft(int v){
    std::lock_guard<std::mutex> lg(m_ft);
    filter_type = v;
  }

public:
  int get_ft(){
    std::lock_guard<std::mutex> lg(m_ft);
    return filter_type;
  }
 
  IListener(){
  }
  
  void onImageResults(std::map<affdex::FaceId, affdex::Face> faces, affdex::Frame image){

    //printf("result start");
    for(auto pair : faces){
      float* face = &(pair.second.emotions.joy);
      /*
      // joy, fear, disgust, sadness, anger, surprise
      cout<< "\n joy: "      <<to_string(face[0]).c_str()
	  << "\n fear: "     << to_string(face[1]).c_str()
	  << "\n disgust: "  << to_string(face[2]).c_str()
	  << "\n sadness: "  << to_string(face[3]).c_str()
	  << "\n anger: "    <<to_string(face[4]).c_str()
	  << "\n surprise: " << to_string(face[5]).c_str()
	  << std::endl;
      */
      myfile << "joy: "<<to_string(face[0]).c_str()<<" anger: "<<to_string(face[4]).c_str()<< " sadness: "<<to_string(face[3]).c_str()<<" surprise: "<<to_string(face[5]).c_str()<<" disgust: "<<to_string(face[2]).c_str()<< " fear: "<<to_string(face[1]).c_str()<<endl;

      int max_index = 0;
      for(int i=1; i<6; i++){
	if(face[i] > face[max_index]){
	  max_index = i;
	}
      }
      
      if(face[max_index] > 1){
	set_ft(max_index+1);
      } else {
	set_ft(0);
      }
    }
  };
  
  void onImageCapture(affdex::Frame image){
    //std::lock_guard<std::mutex> lg(mMutex);
    //printf("onImageCapture\n");
  };
};

/* call back function for new-sample of appsink */
static GstFlowReturn
on_new_sample_from_sink (GstElement * elt, ProgramData * data)
{
  GstSample *sample;
  GstBuffer *app_buffer, *buffer;
  GstFlowReturn ret;

  sample = gst_app_sink_pull_sample (GST_APP_SINK (elt));
  buffer = gst_sample_get_buffer (sample);

  GstMapInfo map;
  gst_buffer_map (buffer, &map, GST_MAP_READ);
  Mat frame(Size(320, 240), CV_8UC3, (char*)map.data, Mat::AUTO_STEP);
  
  // affdex
  affdex::Frame aframe(frame.size().width, frame.size().height, frame.data, Frame::COLOR_FORMAT::BGR);
  data->detector->process(aframe);

  // writing into gstreamer pipeline
  GstMemory *new_mem = gst_allocator_alloc(NULL, map.size, NULL);
  GstMapInfo new_map;
  gst_memory_map(new_mem, &new_map, GST_MAP_WRITE);
  app_buffer = gst_buffer_copy(buffer);
  int index = dynamic_cast<IListener *>(data->detector->getImageListener())->get_ft() - 1;
  if(index != -1){
    int maxVal = pow(2, 8*frame.elemSize1())-1;
    frame =
      img_rgb[index].mul(img_aaa[index], 1.0/(double)maxVal)
      + frame.mul(img_1ma[index], 1.0/(double)maxVal);
  }
  
  memcpy(new_map.data, (uchar*)frame.data, new_map.size);
  gst_buffer_replace_all_memory(app_buffer,new_mem);
  gst_memory_unmap(new_mem, &new_map);
    
  // output
  ret = gst_app_src_push_buffer (GST_APP_SRC (gst_bin_get_by_name (GST_BIN (data->sink), "testsource")), app_buffer);
  // unref
  gst_sample_unref (sample);
  
  return ret;
}
int main (int argc, char *argv[])
{  
  string files[6] = {
    "/home/takumi/affdex-sdk/data/screentones/2happy.png",
    "/home/takumi/affdex-sdk/data/tones/fear.png",
    "/home/takumi/affdex-sdk/data/tones/sidelines.png",
    "/home/takumi/affdex-sdk/data/tones/sadness.png",
    "/home/takumi/affdex-sdk/data/screentones/1anger.png",
    "/home/takumi/affdex-sdk/data/screentones/4shock.png"
  };
  
  for(int i=0; i<6; i++){
    vector<cv::Mat>planes_rgba, planes_rgb, planes_aaa, planes_1ma;

    filtera = imread(files[i],cv::IMREAD_UNCHANGED);
    resize(filtera, filtera, cv::Size(),
	   320.0/filtera.cols,240.0/filtera.rows);
    cv::split(filtera, planes_rgba);
    
    //RGBA画像をRGBに変換

    planes_rgb.push_back(planes_rgba[0]);
    planes_rgb.push_back(planes_rgba[1]);
    planes_rgb.push_back(planes_rgba[2]);
    merge(planes_rgb, img_rgb[i]);
    
    //RGBA画像からアルファチャンネル抽出
    planes_aaa.push_back(planes_rgba[3]);
    planes_aaa.push_back(planes_rgba[3]);
    planes_aaa.push_back(planes_rgba[3]);
    merge(planes_aaa, img_aaa[i]);

    int maxVal = pow(2, 8*img_rgb[i].elemSize1())-1;
    //背景用アルファチャンネル   
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    merge(planes_1ma, img_1ma[i]);

  }


  ProgramData *data = NULL;
  GstElement *testsink = NULL;
  gst_init (&argc, &argv);
  data = g_new0 (ProgramData, 1);
  
  data->loop = g_main_loop_new (NULL, FALSE);

  // affdex!!!
  affdex::path DATA_FOLDER = "/home/takumi/affdex-sdk/data";
  //Initialize camera detector
  //affdex:: FrameDetector detector(1);
  affdex::PhotoDetector detector;
  data->detector = &detector;
  data->detector->setClassifierPath(DATA_FOLDER);
  data->detector->setDetectAllEmotions(true);
  data->detector->setDetectAllEmojis(false);
  data->detector->setDetectAllExpressions(false);
  data->detector->setDetectAllAppearances(false);
  data->detector->setDetectContempt(false);
  data->detector->setDetectValence(false);
  data->detector->setDetectEngagement(false);
  
  std::shared_ptr<affdex::FaceListener> fListen(new FListener());
  std::shared_ptr<affdex::ImageListener> iListen(new IListener());
  data->detector->setFaceListener(fListen.get());
  data->detector->setImageListener(iListen.get());
  
  data->detector->start();

myfile.open ("tracker.txt");
myfile << "Begin User.\n"<<endl;
  
  // No touch!
  // setup src side
  data->source = gst_parse_launch (p_src, NULL);
  if (data->source == NULL) {
    g_print ("Bad source\n");
    g_main_loop_unref (data->loop);
    g_free (data);
    return -1;
  } else {
    // set call back function
    testsink = gst_bin_get_by_name (GST_BIN (data->source), "testsink");
    g_object_set (G_OBJECT (testsink), "emit-signals", TRUE, NULL);
    g_signal_connect (testsink, "new-sample",
              G_CALLBACK (on_new_sample_from_sink), data);
    gst_object_unref (testsink);
  }
  // setup sink side 
  data->sink = gst_parse_launch (p_sink, NULL); 
  if (data->sink == NULL) {
    g_print ("Bad sink\n");
    gst_object_unref (data->source);
    g_main_loop_unref (data->loop);
    g_free (data);
    return -1;
  }

  gettimeofday(&_time ,NULL);

  GstElement *point1_elem;
  GstElement *point2_elem;
  GstElement *point3_elem;

  point1_elem = gst_bin_get_by_name (GST_BIN (data->source), "point1");
  g_signal_connect (point1_elem, "handoff",
		    G_CALLBACK (point1_handoff_handler), data);
  gst_object_unref (point1_elem);

  point2_elem = gst_bin_get_by_name (GST_BIN (data->sink), "point2");
  g_signal_connect (point2_elem, "handoff",
		    G_CALLBACK (point2_handoff_handler), data);
  gst_object_unref (point2_elem);

  point3_elem = gst_bin_get_by_name (GST_BIN (data->sink), "point3");
  g_signal_connect (point3_elem, "handoff",
		    G_CALLBACK (point3_handoff_handler), data);
  gst_object_unref (point3_elem);

  
  /* launching things */
  gst_element_set_state (data->sink, GST_STATE_PLAYING);
  gst_element_set_state (data->source, GST_STATE_PLAYING);
  g_print ("Let's run!\n");
  g_main_loop_run (data->loop);
  g_print ("Going out\n");
  // for finish
  gst_element_set_state (data->source, GST_STATE_NULL);
  gst_element_set_state (data->sink, GST_STATE_NULL);
  gst_object_unref (data->source);
  gst_object_unref (data->sink);
  g_main_loop_unref (data->loop);
  g_free (data);
  myfile.close();
  return 0;
}

