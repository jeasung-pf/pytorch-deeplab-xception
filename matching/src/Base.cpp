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

#include "precomp.hpp"
#include "../include/base.hpp"
#include "../include/all_descriptors.hpp"
#include "../include/general.hpp"

#include <vector>
#include <algorithm>

cv::Matcher::Base::Base(cv::Ptr<cv::Matcher::FeatureParams> _featureParams,
                        cv::Ptr<cv::flann::IndexParams> _indexParams,
                        cv::Ptr<cv::flann::SearchParams> _searchParams,
                        float _filterThreshold) : featureParams(featureParams), indexParams(indexParams), searchParams(searchParams), filterThreshold(_filterThreshold) {

}

//void cv::Matcher::Base::clear() {
//    featureExtractor.release();
//    matcher.release();
//}
//
//
//bool cv::Matcher::Base::train(const cv::Mat &trainData, const cv::Mat &trainMask, std::vector<cv::KeyPoint> &keypoints,
//                              cv::Mat &descriptor) {
//
//    if (trainData.channels() != 1) {
//        cv::cvtColor(trainData, trainData, cv::COLOR_RGB2GRAY);
//    }
//
//    if (trainMask.channels() != 1) {
//        throw cv::Matcher::MatcherException(std::string(
//                "The trainMask entered in the training loop has more than one channel. Check your data.\n"));
//        return false;
//    }
//
//    try {
//        this->featureExtractor->detect(trainData, keypoints, trainMask);
//        this->featureExtractor->compute(trainData, keypoints, descriptor);
//    } catch (std::exception e) {
//        throw cv::Matcher::MatcherException(
//                std::string("Error while computing template features. Consult your implementation.\n"));
//        return false;
//    }
//
//    return true;
//}
//
//bool cv::Matcher::Base::train(const std::vector<cv::Mat> &trainData, const std::vector<cv::Mat> &trainMask) {
//
//    if (trainData.size() != trainMask.size()) {
//        throw cv::Matcher::MatcherException(
//                std::string("The number of trainData doesn't match with the number of trainMask. Check your data.\n"));
//        return false;
//    }
//
//    if (!this->featureExtractor.empty() || !this->templateKeyPoints.empty() || !this->templateDescriptors.empty()) {
//        this->clear();
//    }
//
//    if (this->featureParams.empty()) {
//        throw cv::Matcher::MatcherException(
//                std::string("Missing featureParams in cv::Matcher::Base class.\n"));
//        return false;
//    }
//
//    this->featureExtractor = cv::Matcher::feature_creator::create(*(this->featureParams));
//
//    int listSize = (int) trainData.size();
//    for (int i = 0; i < listSize; i++) {
//        std::vector<cv::KeyPoint> keypoints;
//        cv::Mat descriptor;
//
//        this->train(trainData[i], trainMask[i], keypoints, descriptor);
//        this->templateKeyPoints.push_back(keypoints);
//        this->templateDescriptors.push_back(descriptor);
//    }
//
//    return true;
//}
//
//bool
//cv::Matcher::Base::extract(const cv::Mat &sample, const cv::Mat &sampleMask, std::vector<cv::KeyPoint> &sampleKeyPoints,
//                           cv::Mat &sampleDescriptor) const {
//
//    if (sample.channels() != 1) {
//        cv::cvtColor(sample, sample, cv::COLOR_RGB2GRAY);
//    }
//
//    if (sampleMask.channels() != 1) {
//        throw cv::Matcher::MatcherException(std::string(
//                "The sampleMask entered in the prediction step has more than one channel. Check your data.\n"));
//        return false;
//    }
//
//    if (!this->featureExtractor.empty() || !this->templateKeyPoints.empty() || !this->templateDescriptors.empty()) {
//        throw cv::Matcher::MatcherException(std::string(
//                "This StatModel is not trained yet. Please train the model first and then use it to predict something"));
//        return false;
//    }
//
//    try {
//        this->featureExtractor->detect(sample, sampleKeyPoints, sampleMask);
//        this->featureExtractor->compute(sample, sampleKeyPoints, sampleDescriptor);
//    } catch (std::exception e) {
//        throw cv::Matcher::MatcherException(
//                std::string("Error while computing sample features. Consult your implementation.\n"));
//        return false;
//    }
//
//    return true;
//}

cv::Matcher::Base::~Base() {

}

void cv::Matcher::Base::write(cv::FileStorage *storage, const char *name) const {

}

void cv::Matcher::Base::read(cv::FileStorage *storage, CvFileNode *node) {

}

bool
cv::Matcher::Base::extract(const cv::Mat &sample, const cv::Mat &sampleMask, std::vector<cv::KeyPoint> &sampleKeyPoints,
                           cv::Mat &sampleDescriptor) const {
    return false;
}





