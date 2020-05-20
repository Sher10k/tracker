#ifndef HPP_STEREO_VIZ
#define HPP_STEREO_VIZ

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <map>


class Viz
{
public:
    enum COLOR
    {
        WHITE,
        RED,
        GREEN,
        BLUE,
        YELLOW,
        BROWN,
        BLACK
    };

    Viz( const cv::FileStorage& config )
    {
        config["R"] >> R;
        config["T"] >> T;
        config["P"] >> P;
        config["mtxL"] >> mtx;
        config["distL"] >> dist;

        config["projectL"] >> mtx_new;
        mtx_new = mtx_new.colRange(0, 3);
        dist_new = dist * 0;

        cv::Rodrigues( R*-1, A3D );
        cv::hconcat( A3D, T*-1, A3D );
        std::cout << "A3D: " << A3D << "; " << A3D.type() << "\n";
        std::cout << "P: " << P << "; " << P.type() << "\n";
        // AffineTransform= cv::Affine3d(R, T);

    }

    void draw(
        cv::Mat& img,
        cv::Mat& mask,
        COLOR color )
    {
        std::vector< cv::Mat > channels(3);
        cv::split(img, channels);
        draw( channels, mask, color );

        cv::merge(channels, img);
    }

    void draw(
        std::vector< cv::Mat >& channels,
        cv::Mat& mask,
        COLOR color)
    {
        if ( color == COLOR::WHITE )
        {
            cv::addWeighted( channels[0], 1, mask, 0.2, 0, channels[0] );
            cv::addWeighted( channels[1], 1, mask, 0.2, 0, channels[1] );
            cv::addWeighted( channels[2], 1, mask, 0.2, 0, channels[2] );
        }
        if ( color == COLOR::GREEN )
        {
            cv::addWeighted( channels[1], 1, mask, 0.2, 0, channels[1] );
        }
        if ( color == COLOR::YELLOW )
        {
            cv::addWeighted( channels[1], 1, mask, 0.2, 0, channels[1] );
            cv::addWeighted( channels[2], 1, mask, 0.2, 0, channels[2] );
        }
        if ( color == COLOR::BLUE )
        {
            cv::addWeighted( channels[0], 1, mask, 0.5, 0, channels[0] );
        }
        if ( color == COLOR::RED )
        {
            cv::addWeighted( channels[2], 1, mask, 0.2, 0, channels[2] );
        }
        if ( color == COLOR::BROWN )
        {
            cv::addWeighted( channels[1], 1, mask, 0.1, 0, channels[1] );
            cv::addWeighted( channels[2], 1, mask, 0.2, 0, channels[2] );
        }
        if ( color == COLOR::BLACK )
        {
            cv::addWeighted( channels[0], 1, mask, 0., 0, channels[0] );
            cv::addWeighted( channels[1], 1, mask, 0., 0, channels[1] );
            cv::addWeighted( channels[2], 1, mask, 0., 0, channels[2] );
        }
    }

    void draw(
        cv::Mat& frame,
        cv::Mat& disparity,
        cv::Mat& railway_mask,
        cv::Mat& red_mask,
        cv::Mat& yellow_mask,
        cv::Mat& brown_mask,
        cv::Mat& dst )
    {
        dst = frame.clone();

        std::map< COLOR, cv::Mat& > to_draw = {
            /*
            { COLOR::RED, railway_mask },
            { COLOR::RED, railway_mask },
            { COLOR::RED, railway_mask },
            */
            { COLOR::RED, railway_mask },
            { COLOR::RED, red_mask },
            { COLOR::YELLOW, yellow_mask },
            { COLOR::BROWN, brown_mask },
        };
        draw_masks_colors( dst, to_draw );

        // disparity visualization
	//
	// /
        cv::Mat disparity_viz;
        disparity.convertTo( disparity_viz, CV_8U, 1.0 );
        cv::normalize( disparity_viz, disparity_viz, 0, 255, cv::NORM_MINMAX );
        cv::applyColorMap( disparity_viz, disparity_viz, cv::COLORMAP_RAINBOW );
        cv::resize( disparity_viz, disparity_viz,  { 640, 480 } );

        disparity_viz.copyTo( dst( cv::Rect( 0, 0, 640, 480 ) ) );
    }

    void draw_overrall(
        cv::Mat& img, std::vector< cv::Point3f >& central_points )
    {
        cv::transform( central_points, central_points, P );
        std::vector< cv::Point3f > overall_pts = {
            {  0.00,     0,    0 },
            { -0.76,     0,    0 },
            { -1.85, -0.34,    0 },
            { -1.85, -3.85,    0 },
            { -0.70, -5.30,    0 },
            {  0.70, -5.30,    0 },
            {  1.85, -3.85,    0 },
            {  1.85, -0.34,    0 },
            {  0.76,     0,    0 },
        };

        std::vector< cv::Point3f > obj_pts;

        auto overall_size = overall_pts.size();
        for ( int cp_idx = 0; cp_idx < central_points.size(); cp_idx ++ )
        {
            for ( int i = 0; i < overall_size; i ++ )
            {
                // central_points[cp_idx] = {0, 0, cp_idx * 10};
                obj_pts.push_back( central_points[cp_idx] + overall_pts[i] );
                // obj_pts.push_back( central_points[cp_idx] );
            }
        }

        std::vector< cv::Point2f > img_pts;

        cv::projectPoints( obj_pts, R, T, mtx_new, dist_new, img_pts );

        for ( int i = 0; i < img_pts.size() / overall_size; i++ )
        {
            for ( int j = 0; j < overall_size; j++ )
            {
                auto pt_from = img_pts[i*9+j];
                auto pt_to = img_pts[i*9+(j+1)%9];

                if ( i+1 < img_pts.size()/overall_size)
                {
                    auto pt_next_slice = img_pts[(i+1)*9+j];
                }
                cv::line(img, pt_from, pt_to, {0, 0, 255}, 1);
                cv::circle(img, pt_from, 1, {0, 255, 0}, cv::FILLED);
            }
        }
    }

    void draw_masks_colors( cv::Mat& img, std::map< COLOR, cv::Mat& > data )
    {
        std::vector< cv::Mat > channels(3);
        cv::split(img, channels);

        for ( auto pair : data )
        {
            auto color = pair.first;
            auto mask = pair.second;
            cv::resize( pair.second, pair.second, img.size() );
            if ( pair.second.size().empty() )
            {
                continue;
            }

            draw( channels, mask, color );
        }

        cv::merge(channels, img);
    }
private:
    cv::Mat R, T, A3D, mtx, dist, mtx_new, dist_new, P;
};

#endif
