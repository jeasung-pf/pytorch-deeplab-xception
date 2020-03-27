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

#ifndef OPENCV_MATCHER_BASE_HPP_
#define OPENCV_MATCHER_BASE_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <params.hpp>

namespace cv {
    namespace Matcher {

        class CV_EXPORTS_W Base : public cv::ml::StatModel {
        public:
            Base() {}

            Base(const cv::Ptr<cv::Matcher::FeatureParams> _featureParams,
                 const cv::Ptr<cv::flann::IndexParams> _indexParams,
                 const cv::Ptr<cv::flann::SearchParams> _searchParams,
                 float _filterThreshold = 0.0f);

            virtual ~Base();

//            virtual void clear();

//            virtual int getVarCount() const = 0;
//
//            virtual bool empty() const CV_OVERRIDE;
//
//            virtual bool isTrained() const = 0;
//
//            virtual bool isClassifier() const = 0;
//
//            virtual bool train( const cv::Ptr<cv::ml::TrainData>& trainData, int flags=0 );
//
//            virtual bool train( InputArray samples, int layout, InputArray responses );
//
//            virtual float calcError( const cv::Ptr<cv::ml::TrainData>& data, bool test, OutputArray resp ) const;
//
//            virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;

            virtual void write(cv::FileStorage *storage, const char *name) const;

            virtual void read(cv::FileStorage *storage, CvFileNode *node);

            bool criterion(cv::DMatch element) {
                if (element.distance < filterThreshold) {
                    return true;
                }
                return false;
            }

        protected:

            virtual bool
            extract(const cv::Mat &sample, const cv::Mat &sampleMask, std::vector<cv::KeyPoint> &sampleKeyPoints,
                    cv::Mat &sampleDescriptor) const;

            float filterThreshold;

            cv::Ptr<cv::Matcher::FeatureParams> featureParams;
            cv::Ptr<cv::flann::IndexParams> indexParams;
            cv::Ptr<cv::flann::SearchParams> searchParams;

            cv::Ptr<cv::Feature2D> featureExtractor;
            std::vector<std::vector<cv::KeyPoint>> templateKeyPoints;
            std::vector<cv::Mat> templateDescriptors;

            cv::Ptr<cv::FlannBasedMatcher> matcher;
        };

    }
}

#endif
