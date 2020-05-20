// include stuff
#include "include/utils/argparser.hpp"
#include "include/utils/version.hpp"

// include
#include "include/handler.hpp"

// Computer Vision
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

// UNIX & Linux
#include <sys/types.h>
#include <unistd.h>

// ZCM
#include <zcm/zcm-cpp.hpp>
#include "include/zcm_types/ZcmCameraBaslerJpegFrame.hpp"
#include "include/zcm_types/ZcmService.hpp"
#include "include/zcm_types/ZcmServiceVersion.hpp"
#include "include/zcm_types/ZcmServiceVersionRequest.hpp"

#include "include/default_cfg.hpp"

// STD
#include <iostream>
#include <string>
#include <map>


int main( int argc, const char** argv )
{
    // parse arguments
    auto arguments = Argparser::parse(
        argc, argv,
        {
            { "-v", Argparser::Sufficient },
            { "-h", Argparser::Sufficient },
            { "--save-conf", Argparser::Sufficient },
            { "--print-conf", Argparser::Sufficient },
            { "-p", Argparser::NotSufficient },
            { "-c", Argparser::NotSufficient }
        });

    try {
        arguments.at("--save-conf");
        std::ofstream outfile;
        outfile.open("build/stereo_pair.cfg");
        outfile << default_cfg;
        outfile.close();
        return 0;
    }
    catch (const std::out_of_range& e) {
    }

    try {
        arguments.at("--print-conf");
        std::cout << default_cfg << "\n";
        return 0;
    }
    catch (const std::out_of_range& e) {
    }

    try {
        arguments.at("-c");
        arguments.at("-p");
    }
    catch (const std::out_of_range& e) {
        std::cerr << "Please. Use required arguments -c %config name% -p %pid_file%.\n";
        return 22;
    }

    try {
        arguments.at("-v");
        std::cout << "Version: " << 0.1 << "\n";
        return 0;
    }
    catch (const std::out_of_range& e) {
    }

    try {
        arguments.at("-h");
        std::cout << "OpenCV info:\n";
        std::cout << cv::getBuildInformation() << std::endl;
        return 0;
    }
    catch (const std::out_of_range& e) {
    }

    // create pid file
    std::cout << "Current PID: " << getpid() << "\n";
    std::cout << "Pid path: " << arguments["-p"] << "\n";

    std::ofstream out( arguments["-p"] );
    out << getpid();
    out.close();

    // read configuration file
    std::string filename = arguments[ "-c" ];
    std::cout << "Configuration file is " << filename << "\n";

    cv::FileStorage config( filename, cv::FileStorage::READ );
    try
    {
        config.open( filename, cv::FileStorage::READ );
    }
    catch (const cv::Exception& e)
    {
        std::cout << "Error in configuration parsing.\n";
    }

    // zcm initialization
    zcm::ZCM zcm_in( config["input_addres"] );
    zcm::ZCM zcm_out( config["output_addres"] );

    Handler handlerObject(
        config,
        &zcm_out);

    // subsribing
    zcm_in.subscribe(
        config["camera_channel"], &Handler::handleCamera, &handlerObject
    );
    zcm_in.subscribe(
        config["object_channel"], &Handler::handleTrains, &handlerObject
    );
    zcm_in.run();

    config.release();
}
