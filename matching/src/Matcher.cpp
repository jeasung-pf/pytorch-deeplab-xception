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

cv::Matcher::Matcher::Matcher(const cv::Matcher::FeatureParams &_featureParams,
                              const cv::flann::IndexParams &_indexParams,
                              const cv::flann::SearchParams &_searchParams): Base(_featureParams, _indexParams, _searchParams, 0.0f) {

}

bool cv::Matcher::Matcher::predict(const cv::Mat &sample, const cv::Mat &sampleMask, cv::Mat &transformed,
                                   cv::Matcher::identification_t mode) const {
    if (sample.channels() == 1) {
        cv::cvtColor(sample, sample, cv::COLOR_RGB2GRAY);
    }

    std::vector<cv::KeyPoint> keypoints = this->templateKeyPoints[(int)mode];
    std::vector<cv::KeyPoint> sampleKeyPoints;
    cv::Mat descriptors = this->templateDescriptors[(int)mode];
    cv::Mat sampleDescriptor;

    Base::extract(sample, sampleMask, sampleKeyPoints, sampleDescriptor);

    // Match the given image with the template images
    std::vector<cv::DMatch> matches;
    try {
        this->matcher->match(sampleDescriptor, descriptors, matches);
    } catch (std::exception e) {
        throw cv::Matcher::MatcherException(std::string("Error while matching two images. Consult your implementation.\n"));
    }

    vector<cv::Point2f> image_points, object_points;
    for (auto elem : matches) {
        image_points.push_back(keypoints[elem.queryIdx].pt);
        object_points.push_back(sampleKeyPoints[elem.trainIdx].pt);
    }
    cv::Mat F = cv::findFundamentalMat(object_points, image_points, cv::FM_RANSAC);

    cv::warpPerspective(sample,
            transformed,
            F,
            sample.size(),
            cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar::all(0));

    return true;
}

void cv::Matcher::Matcher::write(CvFileStorage *storage, const char *name) const {

}

void cv::Matcher::Matcher::read(CvFileStorage *storage, CvFileNode *node) {

}

