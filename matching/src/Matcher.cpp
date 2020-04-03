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

#include <Matcher.hpp>


std::string cv::Matcher::Matcher::m_window_image = "Template image window";
std::string cv::Matcher::Matcher::m_window_captured = "The current frame captured";
std::string cv::Matcher::Matcher::m_window_clicked = "The frame captured by mouse click";
std::string cv::Matcher::Matcher::m_window_mask = "Mask created";
std::string cv::Matcher::Matcher::m_window_matching = "Source and Target correspondence";
std::string cv::Matcher::Matcher::m_window_homography = "Projected image using estimated image";

void filterThreshold(std::vector<cv::DMatch> &match, float threshold) {
    auto it = std::remove_if(std::begin(match), std::end(match),
                             [threshold](const cv::DMatch &m) { return m.distance >= threshold; });
    match.erase(it, std::end(match));
}

bool withinCroppedArea(const cv::Point &pt) {
    std::cout << pt.x << " " << pt.y << std::endl;
    if (((int) pt.x >= 50) && ((int) pt.x <= 200) && ((int) pt.y >= 20) && ((int) pt.y <= 50)) {
        return true;
    }
    return false;
}

cv::Matcher::Matcher::Matcher(const cv::Ptr<cv::Matcher::FeatureParams> &_featureParams,
                              const cv::Ptr<cv::flann::IndexParams> &_indexParams,
                              const cv::Ptr<cv::flann::SearchParams> &_searchParams) : Base(_featureParams,
                                                                                            _indexParams, _searchParams,
                                                                                            0.0f) {
    this->brisk = cv::BRISK::create(get_param<int>(*featureParams, "threshold"),
                                    get_param<int>(*featureParams, "octaves"),
                                    get_param<float>(*featureParams, "patternScale"));
}

int cv::Matcher::Matcher::getVarCount() const {
    return 0;
}

bool cv::Matcher::Matcher::empty() const {
    if (brisk.empty() || matcher.empty()) {
        return true;
    }
    return false;
}

bool cv::Matcher::Matcher::isTrained() const {
    return false;
}

bool cv::Matcher::Matcher::isClassifier() const {
    return false;
}

bool cv::Matcher::Matcher::train(const cv::Ptr<cv::ml::TrainData> &trainData, int flags) {
    return StatModel::train(trainData, flags);
}

bool cv::Matcher::Matcher::train(const cv::_InputArray &samples, int layout, const cv::_InputArray &responses) {
    return false;
}

float
cv::Matcher::Matcher::calcError(const cv::Ptr<cv::ml::TrainData> &data, bool test, const cv::_OutputArray &resp) const {
    return StatModel::calcError(data, test, resp);
}

float cv::Matcher::Matcher::predict(const cv::_InputArray &samples, const cv::_OutputArray &results, int flags) const {
    return 0.0;
}

void cv::Matcher::Matcher::write(cv::FileStorage *storage, const char *name) const {
    Base::write(storage, name);
}

void cv::Matcher::Matcher::read(cv::FileStorage *storage, CvFileNode *node) {
    Base::read(storage, node);
}

/// Create window for visualizing process
/// \return true
bool cv::Matcher::Matcher::create_window() {
    try {
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
    } catch (std::exception e) {
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}

/// Estimate a homography between samples image and template image and calibrate samples image accordingly.
/// \param samples : A sample image
/// \param template_image : A template image without any distortion.
/// \param mask : A mask image to be applied in 'samples' image.
/// \param calibrated : A result image containing calibrated pixels.
/// \param threshold : Threshold to be applied in filtering matches between images.
/// \return true if nothing goes wrong.
bool cv::Matcher::Matcher::train(const cv::_InputArray &samples, const cv::_InputArray &template_image,
                                 const cv::_InputArray &mask, const cv::_OutputArray &calibrated, float threshold) {
    if (!samples.sameSize(mask)) {
        CV_Error(CV_StsBadArg, "The size of the sample image and mask doesn't match.");
    }

    std::vector<std::vector<cv::DMatch>> matches;
    cv::Mat matched_image;

    samples.copyTo(calibrated, mask);
    if (window_created) {
        std::cout << "Showing calibrated image" << std::endl;
        cv::imshow(m_window_clicked, calibrated);
    }

    std::cout << "Computing features..." << std::endl;
    brisk->detectAndCompute(samples, cv::noArray(), samples_key_points, samples_descriptors, false);
    brisk->detectAndCompute(template_image, cv::noArray(), template_key_points, template_descriptors, false);
    matcher->knnMatch(samples_descriptors, template_descriptors, matches, 1);

    std::cout << "Threshold value " << threshold << std::endl;
    for (std::vector<cv::DMatch> &elem: matches) {
        filterThreshold(elem, threshold);
    }
    std::cout << "Drawing matches..." << std::endl;
    cv::drawMatches(calibrated, samples_key_points, template_image, template_key_points, matches, matched_image);

    template_corners.clear();
    samples_corners.clear();
    for (auto &match: matches) {
        if (match.size() == 0)
            continue;

        for (auto &elem : match) {
            if (withinCroppedArea(template_key_points[elem.trainIdx].pt)) {
                template_corners.push_back(template_key_points[elem.trainIdx].pt);
                samples_corners.push_back(samples_key_points[elem.queryIdx].pt);
            }
        }
    }
    std::cout << "Computing homography..." << std::endl;
    std::cout << "The number of corners is " << template_corners.size() << std::endl;
    cv::Mat H = cv::findHomography(samples_corners, template_corners, cv::RANSAC);
//    cv::Mat F = cv::findFundamentalMat(frame_corners, original_image_corners, cv::FM_RANSAC);

    // do perspective transformation
    cv::warpPerspective(calibrated, calibrated, H, cv::Size());
    std::cout << "Image warped..." << std::endl;

    if (window_created) {
        cv::imshow(m_window_homography, calibrated);
        cv::imshow(m_window_matching, matched_image);
    }
    return true;
}




