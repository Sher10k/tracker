#include "handler.hpp"

// ZCM
#include <zcm/zcm-cpp.hpp>
#include "zcm_types/ZcmCameraBaslerJpegFrame.hpp"
#include "zcm_types/ZcmRailDetectorMask.hpp"
#include "zcm_types/ZcmTextData.hpp"

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
#include <sstream>


Handler::Handler(
        const cv::FileStorage& config,
        zcm::ZCM* zcm_out,
        bool verbose )
    :
        zcm_out( zcm_out ),
        zcm_viz( zcm_viz ),
        tracker( NearestNeighborDistanceMetric( "euclidean", 1.0e+08f, 100 ), 0.8f, 10, 2 )
{
    // application logic parameters
    float period_s;
    config["period"] >> period_s;
    periodic_ts = int64_t( periodic_ts*std::pow( 10, 6 ) );
};

void Handler::handleCamera(
    const zcm::ReceiveBuffer*,
    const std::string& channel,
    const ZcmCameraBaslerJpegFrame *msg )
{
    clock_t start_time = clock();
    clock_t init_time = clock();
    std::stringstream profile_stream;
    profile_stream << "Init handling timestamp: "
                   << int( 1.0 * init_time / CLOCKS_PER_SEC * std::pow( 10, 6 ) ) << "\n";

    // contain frame as last left or last right frame
    std::vector< char > jpeg_buf;
    jpeg_buf.assign(
        msg->jpeg.data(),
        msg->jpeg.data() + msg->jpeg.size() );

    last_frame = cv::imdecode( jpeg_buf, cv::IMREAD_GRAYSCALE );
    int64_t last_railway_ts = msg->service.u_timestamp;

    if ( last_frame.size() == cv::Size( 0, 0 ) ) return;

    mtx = cv::Mat(3, 3, CV_32F, (void*)msg->info.calibrating_params.cam_mtx);
    dist = cv::Mat(1, 5, CV_32F, (void*)msg->info.calibrating_params.distCoeff);
    tvec = cv::Mat(1, 3, CV_32F, (void*)msg->info.calibrating_params.tvec);
    rvec = cv::Mat(1, 3, CV_32F, (void*)msg->info.calibrating_params.rvec);

    const int roi_left = -10;
    const int roi_right = 10;
    const float roi_near = 14.0;
    const int length = 100;

    // массив точек объекта objectPoints 3D
    std::vector< cv::Point3f > objectPoints;
    objectPoints.push_back( cv::Point3f(roi_left, 0, roi_near) );
    objectPoints.push_back( cv::Point3f(roi_right, 0, roi_near) );
    objectPoints.push_back( cv::Point3f(roi_left, 0, roi_near + length) );
    objectPoints.push_back( cv::Point3f(roi_right, 0, roi_near + length) );

    std::vector< cv::Point2f > ground_pts;
    ground_pts.push_back( cv::Point2f(roi_left, roi_near) );
    ground_pts.push_back( cv::Point2f(roi_right, roi_near) );
    ground_pts.push_back( cv::Point2f(roi_left, roi_near + length) );
    ground_pts.push_back( cv::Point2f(roi_right, roi_near + length) );

    // cv::projectPoints -> массив точек на изображении imgPoints 2D
    std::vector< cv::Point2f > imgPoints;

    cv::projectPoints(objectPoints, rvec, tvec, mtx, dist, imgPoints);

    H = cv::getPerspectiveTransform(imgPoints, ground_pts);

    return;
};

ZcmPoint create_zcm_point( float x, float y, float z )
{
    ZcmPoint retval;
    retval.x = x;
    retval.y = y;
    retval.z = z;

    return retval;
}

void Handler::handleTrains(
    const zcm::ReceiveBuffer*,
    const std::string& channel,
    const ZcmObjectList *msg )
{
    frame_idx++;
    std::cout << "\nProcessing frame: " << frame_idx << " -------------------------------------------------- ///" << "\n";

    std::cout << "Train's count: " << msg->detected_objects.size() << "\n";
    last_objects = *msg;

    std::vector< cv::Point3f > object_points3D;
    for ( auto obj : last_objects.detected_objects )
    {
        auto pt = obj.bounding_box.center;
        float width = obj.bounding_box.width;
        float height = obj.bounding_box.height;

        object_points3D.push_back( {-1*float(pt.y) - width/2, 1*float(pt.z), 1*float(pt.x)} );
        object_points3D.push_back( {-1*float(pt.y) - width/2, 1*float(pt.z) - height, 1*float(pt.x)} );
        object_points3D.push_back( {-1*float(pt.y) + width/2, 1*float(pt.z) - height, 1*float(pt.x)} );
        object_points3D.push_back( {-1*float(pt.y) + width/2, 1*float(pt.z), 1*float(pt.x)} );
    }

    if ( mtx.size() != cv::Size( 3, 3 ) or object_points3D.size() == 0 ) 
    {
        // Производим предсказание
        tracker.predict();
        // Отображаем треки на кадре даже если не пришли объекты
        view_track( msg->service.u_timestamp );
        return;
    }

    // пересчитываем 3D объекты в 2D проекции на кадр
    std::vector< cv::Point2f > projected_points;
    cv::projectPoints( object_points3D, rvec, tvec, mtx, dist, projected_points);
    // std::cout << "Projected points:\n";

    std::vector< Detection > detections;
    for ( int i = 0; i < projected_points.size(); i += 4 )
    {
        auto pt = projected_points[i];
        // std::cout << pt << "\n";
        int height = projected_points[i+3].y-projected_points[i+1].y;
        int width = projected_points[i+3].x-projected_points[i+1].x;
        int x = projected_points[i+1].x;
        int y = projected_points[i+1].y;

        if ( y+height >= last_frame.size().height or x+width >= last_frame.size().width or x < 0 or y < 0 ) continue;

        //  преобразуем фрагмент изображения объекта в features
        cv::Mat object_img;
        last_frame(cv::Rect(x, y, width, height)).convertTo(object_img, CV_32F, 1.0/255);
        cv::resize(object_img, object_img, cv::Size(64, 64));
        std::vector< float > object_features( object_img.data, object_img.data + object_img.total() );
        Detection detection(cv::Rect2f(x, y, width, height), 0.5, object_features);
        detections.push_back( detection );

        // отрисовываем спроецированные объекты на кадре
        cv::line( last_frame, cv::Point(x,y), cv::Point(x+width,y), 255, 4 );
        cv::line( last_frame, cv::Point(x+width,y), cv::Point(x+width,y+height), 255, 4 );
        cv::line( last_frame, cv::Point(x+width,y+height), cv::Point(x,y+height), 255, 4 );
        cv::line( last_frame, cv::Point(x,y+height), cv::Point(x,y), 255, 4 );
    }

    // Делаем предсказание на 1 кадр и обновляем треки
    tracker.predict();
    tracker.update( detections );

    // Отрисовыеваем треки после предсказания и обновления
    view_track( msg->service.u_timestamp );

    // std::vector< std::vector< float > > results;
    std::vector< cv::Point2f > tracked_image_pts;
    std::cout << "Track size: " << tracker.tracks.size() << "\n";
    for ( auto track : tracker.tracks )
    {
        if ( (not track.is_confirmed()) || (track.time_since_update > 1) )
            continue;
        cv::Rect2f bbox = track.to_tlwh();
        // results.push_back( { float(1), 
        //                      float(track.track_id), 
        //                      bbox.x, 
        //                      bbox.y, 
        //                      bbox.width, 
        //                      bbox.height,
        //                      1, -1, -1, -1} );
        std::cout << "bbox: \n" << bbox << "\n";

        // std::vector< cv::Point2f > pts = {
        //     {bbox.x, bbox.y},
        //     {bbox.x+bbox.width, bbox.y},
        //     {bbox.x+bbox.width, bbox.y+bbox.height},
        //     {bbox.x, bbox.y+bbox.height}
        // };

        // 3 точки основания объекта
        tracked_image_pts.push_back( {bbox.x, bbox.y+bbox.height} );
        tracked_image_pts.push_back( {bbox.x+bbox.width*0.5f, bbox.y+bbox.height} );
        tracked_image_pts.push_back( {bbox.x+bbox.width, bbox.y+bbox.height} );
    }

    if ( tracked_image_pts.size() == 0 ) return;
    std::vector< cv::Point2f > tracked_object_pts;
    // std::cout << "H: \n" << H << "\n";
    cv::perspectiveTransform( tracked_image_pts, tracked_object_pts, H );
        
    int32_t obj_count;
    last_objects.obj_count = tracked_object_pts.size() / 3;
    last_objects.detected_objects.clear();

    for ( int i = 0; i < tracked_object_pts.size(); i+= 3 )
    {
        ZcmObject obj;
        obj.detection_prob = 1.0;
        obj.recognition_prob = 1.0;

        obj.id_type = 5;
        obj.label_type = "CAR";

        ZcmObjectGeometry box;
        box.top_left = create_zcm_point( tracked_object_pts[i].y, tracked_object_pts[i].x, 1.0 );
        box.bottom_left = create_zcm_point( tracked_object_pts[i].y, tracked_object_pts[i].x, 0.0 );
        box.top_right = create_zcm_point( tracked_object_pts[i+2].y, tracked_object_pts[i+2].x, 1.0 );
        box.bottom_right = create_zcm_point( tracked_object_pts[i+2].y, tracked_object_pts[i+2].x, 0.0 );

        box.center = create_zcm_point( tracked_object_pts[i+1].y, tracked_object_pts[i+1].x, 0.5 );
        box.bottom_center = create_zcm_point( tracked_object_pts[i+1].y, tracked_object_pts[i+1].x, 0 );

        box.width = tracked_object_pts[i+2].x-tracked_object_pts[i+0].x;
        box.height = 1.0;

        obj.bounding_box = box;
        last_objects.detected_objects.push_back(obj);

        // std::cout << "tracked_object_pts_" << int(i/3) << ": " 
        //           << tracked_object_pts[i] << " " 
        //           << tracked_object_pts[i+1] << " " 
        //           << tracked_object_pts[i+2] << "\n";
    }

    zcm_out->publish( channel+"FILT", &last_objects);
};

// отрисовка треков при поступлении или отсутствии объектов
void Handler::view_track( int64_t timestamp )
{
    if ( !last_frame.empty() )
    {
        cv::Mat temp;
        last_frame.copyTo( temp );
        cv::cvtColor( temp, temp, cv::COLOR_GRAY2BGR );

        for ( auto track : tracker.tracks )
        {
            if ( track.is_tentative() )
            {
                cv::Rect2f bbox = track.to_tlwh();
                std::vector< cv::Point2f > pts = {
                    {bbox.x, bbox.y},
                    {bbox.x+bbox.width, bbox.y},
                    {bbox.x+bbox.width, bbox.y+bbox.height},
                    {bbox.x, bbox.y+bbox.height}
                };
                
                // отрисовываем голубым треки, их ID : age : time_since_update
                cv::Scalar Color = cv::Scalar(255,100,0);
                cv::line( temp, pts[0], pts[1], Color, 1 );
                cv::line( temp, pts[1], pts[2], Color, 1 );
                cv::line( temp, pts[2], pts[3], Color, 1 );
                cv::line( temp, pts[3], pts[0], Color, 1 );
                cv::putText( temp, std::to_string(track.track_id) + " : " + std::to_string(track.age) + " : " + std::to_string(track.time_since_update), 
                             pts[0], cv::FONT_HERSHEY_SIMPLEX, 1, Color, 2, cv::LINE_8 );
            }
            else if ( track.is_confirmed() )// || (track.time_since_update > 1) )
            {
                cv::Rect2f bbox = track.to_tlwh();
                // std::cout << "bbox: \n" << bbox << "\n";

                std::vector< cv::Point2f > pts = {
                    {bbox.x, bbox.y},
                    {bbox.x+bbox.width, bbox.y},
                    {bbox.x+bbox.width, bbox.y+bbox.height},
                    {bbox.x, bbox.y+bbox.height}
                };
                
                // отрисовываем желтым треки, их ID : age : time_since_update
                cv::Scalar Color = cv::Scalar(0, int((255 / tracker.max_age) * track.time_since_update), 255);
                cv::line( temp, pts[0], pts[1], Color, 1 );
                cv::line( temp, pts[1], pts[2], Color, 1 );
                cv::line( temp, pts[2], pts[3], Color, 1 );
                cv::line( temp, pts[3], pts[0], Color, 1 );
                cv::putText( temp, std::to_string(track.track_id) + " : " + std::to_string(track.age) + " : " + std::to_string(track.time_since_update), 
                             pts[0], cv::FONT_HERSHEY_SIMPLEX, 1, Color, 2, cv::LINE_8 );
            }
        }
        cv::putText( temp, std::to_string(frame_idx), cv::Point(0, 100), 
                     cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 3, cv::LINE_8 );
        cv::resize( temp, temp, last_frame.size()/2 );
        cv::imshow( "track", temp );
        cv::imwrite( "../zcm_files/outimg/outimg_" + std::to_string(timestamp) + ".png", temp );
        cv::waitKey(10);
    }
}


// void Handler::publish_info( std::string channel, std::string msg )
// {
//     // ZcmTextData json;
//     // json.name = std::string( "zones" );
//     // json.format = std::string( "json" );
//     // json.data = msg;

//     // json.service = ZcmService();
//     // json.service.processing_time = 0;
//     // json.service.u_timestamp = sync_pair.timestamp;

//     // zcm_viz->publish( channel, &json );
// }

// void Handler::publish_jpeg( std::string channel, cv::Mat& img )
// {
//     // ZcmCameraBaslerJpegFrame output;
//     // cv::imencode(".jpg", img, output.jpeg);
//     // output.jpeg_size = output.jpeg.size();
//     // output.service = ZcmService();
//     // output.service.processing_time = 0;
//     // output.service.u_timestamp = sync_pair.timestamp;

//     // zcm_viz->publish( channel, &output );
// }

// void Handler::publish( int32_t start_time, std::string channel, ZcmFromOduObstacles *msg )
// {
//     // msg->service.u_timestamp = sync_pair.timestamp;
//     // msg->service.processing_time = int32_t( 1000000*( 1.0*clock()-start_time ) / CLOCKS_PER_SEC );

//     // zcm_out->publish( channel, msg );
// };
