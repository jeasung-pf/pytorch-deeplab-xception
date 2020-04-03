/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2020 JeaSung (jeasung@peoplefund.co.kr). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <stdlib.h>

#include <grpcpp/grpcpp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Matcher.hpp"

#ifdef BAZEL_BUILD
#include "matching.grpc.pb.h"
#else

#include "matching.grpc.pb.h"

#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using protocol::Feature;
using protocol::Features;
using protocol::Response;
using protocol::Responses;

static void help(char **argv) {
    // print a welcome message, and the OpenCV version
    std::cout
            << "\nThis is a demo program shows how perspective transformation applied on an image, \nUsing OpenCV version "
            << CV_VERSION << std::endl;

    std::cout << "\nUsage:\n" << argv[0] << " [image_name -- Default right.jpg] [threshold] [reprojection error]\n"
              << std::endl;

    std::cout << "\nHot keys: \n"
                 "\tESC, q - quit the program\n"
                 "\tr - change order of points to rotate transformation\n"
                 "\tc - delete selected points\n"
                 "\ti - change order of points to inverse transformation \n"
                 "\nUse your mouse to select a point and move it to see transformation changes" << std::endl;
}

class MatchingClient {
public:
    MatchingClient(std::shared_ptr <Channel> channel) : stub_(protocol::Matching::NewStub(channel)) {

        vstream.open(-1);
        if (!vstream.isOpened()) {
            CV_Error(CV_StsInternal, "Error while creating a video capture device.");
        }

        // create a descriptor matcher for list of orb feature.
        featureParams = new cv::Matcher::BRISKFeatureParams();
        indexParams = new cv::flann::LshIndexParams(10, 10, 2);
        searchParams = new cv::flann::SearchParams();

        // create an instance of matcher class.
        m = new cv::Matcher::Matcher(featureParams, indexParams, searchParams);
        try {
            m->create_window();
            m->window_created = true;
        } catch (std::exception e) {
            std::cout << e.what() << std::endl;
            m->window_created = false;
        }

        cv::setMouseCallback(m->m_window_captured, onMouse, 0);
        cv::setMouseCallback(m->m_window_clicked, onMouseMask, 0);
    }

    void RunClient() {
        bool endProgram = false;
        while (!endProgram) {
            vstream >> captured;
            cv::imshow(cv::Matcher::Matcher::m_window_captured, captured);

            if (dragging) {
                // TODO. inlier analysis is required.
                std::cout << "Calling service..." << std::endl;

                cv::Mat calibrated;
                std::string res;

                res = Calibrate(clicked, mask, calibrated);
                std::cout << res << std::endl;

                dragging = false;
                std::cout << "Dragging: " << dragging << std::endl;
            } else if (validation_needed) {
                captured.copyTo(clicked);
                cv::imshow(cv::Matcher::Matcher::m_window_clicked, clicked);
                validation_needed = false;
            }

            char c = (char) cv::waitKey(10);

            if ((c == 'q') | (c == 'Q') | (c == 27)) {
                endProgram = true;
            }

            if ((c == 'c') | (c == 'C')) {
                roi_corners.clear();
            }

            if ((c == 'r') | (c == 'R')) {
                roi_corners.push_back(roi_corners[0]);
                roi_corners.erase(roi_corners.begin());
            }

            if ((c == 'i') | (c == 'I')) {
                std::swap(roi_corners[0], roi_corners[1]);
                std::swap(roi_corners[2], roi_corners[3]);
            }

        }
    }

    std::string Calibrate(cv::InputArray clicked, cv::InputArray mask, cv::OutputArray calibrated) {
        // Data we are sending to the server.
        Feature feature;
        std::vector<uchar> image, image_segmentation_class;

        std::cout << "Encoding image..." << std::endl;
        try {
            cv::imencode(".jpg", clicked, image);
            cv::imencode(".jpg", mask, image_segmentation_class);
        } catch (std::exception e) {
            std::cout << e.what() << std::endl;
            std::cout << "Error while encoding image..." << std::endl;
        }
        std::string image_encoded(image.begin(), image.end()),
                image_segmentation_class_encoded(image_segmentation_class.begin(), image_segmentation_class.end());

        std::cout << "Creating request..." << std::endl;
        feature.set_image_encoded(image_encoded);
        feature.set_image_filename("image.jpg");
        feature.set_image_format(".jpg");
        feature.set_image_height(clicked.size().height);
        feature.set_image_width(clicked.size().width);
        feature.set_image_segmentation_class_encoded(image_segmentation_class_encoded);
        feature.set_image_segmentation_class_format(".jpg");

        // Container for the data we expect from the server.
        Response response;

        // Context for the client. It could be used to convey extra information to the server and/or tweak certain
        // RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status;
        try {
            std::cout << "Sending request..." << std::endl;
            status = stub_->calibrate(&context, feature, &response);
        } catch (std::exception e) {
            std::cout << e.what() << std::endl;
            std::cout << "Error while calling remote procedure." << std::endl;
        }

        // Act upon its status.
        if (status.ok()) {
            return "OK";
        } else {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return "RPC failed";
        }
    }

private:

    static void onMouse(int event, int x, int y, int, void *) {

        if (event == cv::EVENT_LBUTTONDOWN) {
            std::cout << "(" << x << ", " << y << ")" << std::endl;
            validation_needed = true;
        }
    }

    static void onMouseMask(int event, int x, int y, int flags, void *userdata) {

        if (event == cv::EVENT_LBUTTONDOWN) {
            if (roi_corners.size() > 2)
                cv::line(captured, cv::Point(x, y), roi_corners[roi_corners.size() - 1], cv::Scalar(0, 255, 0));

            roi_corners.push_back(cv::Point(x, y));
            std::cout << "Point(" << x << ", " << y << ") appended" << std::endl;

            imshow(cv::Matcher::Matcher::m_window_captured, captured);
        }

        if (event == cv::EVENT_RBUTTONDOWN) {
            mousedown = false;
            if (roi_corners.size() > 2) {
                // create mask exactly the same size as the captured image
                mask = cv::Mat(captured.size(), CV_8UC1);
                mask = 0;
                // draw contours
                contours.push_back(roi_corners);
                drawContours(mask, contours, 0, cv::Scalar(255), -1);
                cv::Mat masked(captured.size(), CV_8UC3, cv::Scalar(255, 255, 255));
                mask.copyTo(masked, mask);
                imshow(cv::Matcher::Matcher::m_window_mask, masked);
                roi_corners.clear();
                dragging = true;
            }
        }
    }

public:
    cv::Ptr <cv::Matcher::Matcher> m;

public:
    static int mousedown;
    static bool dragging;
    static bool validation_needed;


private:
    std::unique_ptr <protocol::Matching::Stub> stub_;

    cv::VideoCapture vstream;
    cv::Ptr <cv::Matcher::FeatureParams> featureParams;
    cv::Ptr <cv::flann::IndexParams> indexParams;
    cv::Ptr <cv::flann::SearchParams> searchParams;

private:
    static cv::Mat image, captured, clicked, mask, matching;
    static std::vector <cv::Point> roi_corners;
    static std::vector <std::vector<cv::Point>> contours;
};

int MatchingClient::mousedown = 0;
bool MatchingClient::dragging = false;
bool MatchingClient::validation_needed = false;

cv::Mat MatchingClient::image,
        MatchingClient::captured,
        MatchingClient::clicked,
        MatchingClient::mask,
        MatchingClient::matching;
std::vector <cv::Point> MatchingClient::roi_corners;
std::vector <std::vector<cv::Point>> MatchingClient::contours;

int main(int argc, char **argv) {
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint specified by
    // the argument "--target=" which is the only expected argument.
    // We indicate that the channel isn't authenticated (use of
    // InsecureChannelCredentials()).
    std::string target_str;
    std::string arg_str("--target");
    if (argc > 1) {
        std::string arg_val = argv[1];
        size_t start_pos = arg_val.find(arg_str);
        if (start_pos != std::string::npos) {
            start_pos += arg_str.size();
            if (arg_val[start_pos] == '=') {
                target_str = arg_val.substr(start_pos + 1);
            } else {
                std::cout << "The only correct argument syntax is --target=" << std::endl;
                return 0;
            }
        } else {
            std::cout << "The only acceptable argument is --target=" << std::endl;
            return 0;
        }
    } else {
        target_str = "localhost:50052";
    }

    std::cout << "Connecting to server located at " + target_str << std::endl;
    MatchingClient matcher(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    matcher.RunClient();

    return 0;
}







