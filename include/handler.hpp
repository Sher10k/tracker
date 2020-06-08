#ifndef HPP_HANDLER
#define HPP_HANDLER

// ZCM
#include <zcm/zcm-cpp.hpp>
#include "zcm_types/ZcmCameraBaslerJpegFrame.hpp"
#include "zcm_types/ZcmRailDetectorMask.hpp"
#include "zcm_types/ZcmObjectList.hpp"
#include "zcm_types/ZcmFromOduObstacles.hpp"

// Tracker
#include "vtracker/header/tracker.h"

// Computer vision
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <string>
#include <iostream>
#include <ctime>
#include <map>
#include <vector>          // std::priority_queue


class Handler
{
public:
    Handler(
        const cv::FileStorage& config,
        zcm::ZCM* zcm_out,
        bool verbose=false);

    void handleCamera(
        const zcm::ReceiveBuffer*,
        const std::string& channel,
        const ZcmCameraBaslerJpegFrame *msg);

    void handleTrains(
        const zcm::ReceiveBuffer*,
        const std::string& channel,
        const ZcmObjectList *msg);

    void publish(
        int32_t start_time, std::string channel, ZcmFromOduObstacles *msg );

    void publish_jpeg( std::string channel, cv::Mat& img );
    void publish_info( std::string channel, std::string json );

    // void publish_json(

    void publish_clusters(
        ZcmObjectList *msg );

private:
    int64_t periodic_ts;

    int64_t last_left_ts;
    int64_t last_right_ts;
    int64_t last_railway_ts;

    cv::Mat last_left_img, last_right_img, last_railway_mask;

    zcm::ZCM* zcm_out;
    zcm::ZCM* zcm_viz;

    cv::Mat mtx, dist, rvec, tvec;
    cv::Mat last_frame;
    cv::Mat H;
    ZcmObjectList last_objects;

    Tracker tracker;

    int frame_idx = 0;

    void view_track( int64_t timestamp );
};

#endif
