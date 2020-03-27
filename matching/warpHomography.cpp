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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <thread>
#include <unistd.h>
#include <stdlib.h>


static void help(char **argv) {
    // print a welcome message, and the OpenCV version
    std::cout
            << "\nThis is a demo program shows how perspective transformation applied on an image, \nUsing OpenCV version "
            << CV_VERSION << std::endl;

    std::cout << "\nUsage:\n" << argv[0] << " [image_name -- Default right.jpg] [threshold] [reprojection error]\n" << std::endl;

    std::cout << "\nHot keys: \n"
                 "\tESC, q - quit the program\n"
                 "\tr - change order of points to rotate transformation\n"
                 "\tc - delete selected points\n"
                 "\ti - change order of points to inverse transformation \n"
                 "\nUse your mouse to select a point and move it to see transformation changes" << std::endl;
}

static void onMouse(int event, int x, int y, int, void *);

static void onMouseMask(int event, int x, int y, int flags, void *userdata);

cv::Mat warping(cv::Mat image, cv::Size warped_image_size, std::vector<cv::Point2f> srcPoints,
                std::vector<cv::Point2f> dstPoints);

std::string windowTitle = "Perspective Transformation Demo";
std::string labels[4] = {
        "TL", "TR", "BR", "BL"
};
std::vector<cv::Point> roi_corners;
std::vector<std::vector<cv::Point>> contours;
int roiIndex;
bool dragging = false;
int selected_corner_index = 0;
bool validation_needed = false;

static cv::Mat image, captured, clicked, mask, matching, homography;

static std::string m_window_image = "Template image window",
        m_window_captured = "The current frame captured",
        m_window_clicked = "The frame captured by mouse click",
        m_window_mask = "Mask created",
        m_window_matching = "Source and Target correspondence",
        m_window_homography = "Projected image using estimated image";

bool createWindow(void) {
    cv::namedWindow(m_window_image, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(m_window_captured, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(m_window_clicked, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(m_window_mask, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(m_window_matching, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(m_window_homography, CV_WINDOW_AUTOSIZE);

    cv::moveWindow(m_window_image, 0, 0);
    cv::moveWindow(m_window_captured, 640, 0);
    cv::moveWindow(m_window_clicked, 1280, 0);
    cv::moveWindow(m_window_mask, 0, 540);
    cv::moveWindow(m_window_matching, 640, 540);
    cv::moveWindow(m_window_homography, 1280, 540);

    return true;
}

bool withinCroppedArea(const cv::Point &pt) {
    if (((int)pt.x >= 50) && ((int)pt.x <= 200) && ((int)pt.y >= 20) && ((int)pt.y <= 50)) {
        return true;
    }
    return false;
}

int main(int argc, char *argv[]) {

    float threshold = std::stof(std::string(argv[2]));
    float reprojection_error = std::stof(std::string(argv[3]));

    help(argv);

    cv::CommandLineParser parser(argc, argv, "{@input| right.jpg |}");

    std::string filename = cv::samples::findFile(parser.get<std::string>("@input"));
    cv::VideoCapture vstream(-1);
    cv::Mat original_image = cv::imread(filename);
    if (!vstream.isOpened()) {
        CV_Error(CV_StsInternal, "Error while creating a video capture device.");
    }

    float original_image_cols = (float) original_image.cols;
    float original_image_rows = (float) original_image.rows;

    // TODO. find a list of ORB keypoints
    cv::Ptr<cv::BRISK> orb = cv::BRISK::create();
    std::vector<cv::KeyPoint> original_image_key_points, frame_key_points;
    cv::Mat original_image_descriptors, frame_descriptors;
    std::vector<cv::Point2f> original_image_corners, frame_corners;

    // create a descriptor matcher for list of orb feature.
    cv::Ptr<cv::flann::LshIndexParams> indexParams = new cv::flann::LshIndexParams(10, 10, 2);
    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher(indexParams, new cv::flann::SearchParams());
    std::vector<std::vector<cv::DMatch>> matches;

    createWindow();

    cv::imshow(m_window_image, original_image);
    cv::setMouseCallback(m_window_captured, onMouse, 0);
    cv::setMouseCallback(m_window_clicked, onMouseMask, 0);

    bool endProgram = false;
    while (!endProgram) {
        vstream >> captured;
        cv::imshow(m_window_captured, captured);

        if (dragging) {
            // TODO. inlier analysis is required.
            cv::Mat clicked_masked, original_image_threshold, clicked_masked_threshold;
            try {
                std::vector<std::vector<cv::DMatch>> filtered;
                clicked.copyTo(clicked_masked, mask);
                // Apply adaptive thresholding
//                cv::cvtColor(original_image, original_image_threshold, cv::COLOR_RGB2GRAY);
//                cv::cvtColor(clicked_masked, clicked_masked_threshold, cv::COLOR_RGB2GRAY);
//                cv::adaptiveThreshold(original_image_threshold, original_image_threshold, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
//                cv::adaptiveThreshold(clicked_masked_threshold, clicked_masked_threshold, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
//
//                if (!clicked_masked_threshold.data) {
//                    CV_Error(CV_StsNullPtr, "Image is null");
//                }

//                cv::imshow(m_window_image, original_image_threshold);
//                cv::imshow(m_window_clicked, clicked_masked_threshold);
                std::cout << "Computing features..." << std::endl;
                orb->detectAndCompute(original_image, cv::noArray(), original_image_key_points, original_image_descriptors, false);
                orb->detectAndCompute(clicked_masked, cv::noArray(), frame_key_points, frame_descriptors, false);
                matcher->knnMatch(frame_descriptors, original_image_descriptors, matches, 1);
                std::cout << "Threshold value " << threshold << std::endl;
                for (auto elem: matches) {
                    std::vector<cv::DMatch> filtered_elem;
                    for (auto match: elem) {
                        std::cout << match.distance << std::endl;
                        if ((float)match.distance < threshold && withinCroppedArea(original_image_key_points[match.trainIdx].pt)) {
                            filtered_elem.push_back(match);
                        }
                    }

                    if (filtered_elem.size() != 0) {
                        filtered.push_back(filtered_elem);
                    }
                }
                std::cout << filtered.size() << std::endl;

                std::vector<std::vector<cv::DMatch>> matches_top_n(filtered.begin(), filtered.end());
                cv::drawMatches(clicked_masked, frame_key_points, original_image, original_image_key_points, matches_top_n, matching);

                for (auto &match: filtered) {
                    for (auto &elem : match) {
                        original_image_corners.push_back(original_image_key_points[elem.trainIdx].pt);
                        frame_corners.push_back(frame_key_points[elem.queryIdx].pt);
                    }
                }
                cv::Mat H = cv::findHomography(frame_corners, original_image_corners, cv::RANSAC, reprojection_error);
                cv::Mat F = cv::findFundamentalMat(frame_corners, original_image_corners, cv::FM_RANSAC);
                // do perspective transformation
                cv::Mat warped_image;
                cv::warpPerspective(clicked_masked, warped_image, H, cv::Size());

                cv::imshow(m_window_homography, warped_image);
                cv::imshow(m_window_matching, matching);
            } catch (std::exception e) {
                std::cout << e.what() << std::endl;
            }
            dragging = false;
            std::cout << "Dragging: " << dragging << std::endl;
        } else if (validation_needed) {
            captured.copyTo(clicked);
            cv::imshow(m_window_clicked, clicked);
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

    orb.release();
    matcher.release();

    return 0;
}

int mousedown = false;

static void onMouseMask(int event, int x, int y, int flags, void *userdata) {

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (roi_corners.size() > 2)
            cv::line(captured, cv::Point(x, y), roi_corners[roi_corners.size() - 1], cv::Scalar(0, 255, 0));

        roi_corners.push_back(cv::Point(x, y));
        std::cout << "Point(" << x << ", " << y << ") appended" << std::endl;

        imshow(m_window_captured, captured);
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
            imshow(m_window_mask, masked);
            dragging = true;
        }
    }
}


static void onMouse(int event, int x, int y, int, void *) {

    if (event == cv::EVENT_LBUTTONDOWN) {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
        validation_needed = true;
    }
}