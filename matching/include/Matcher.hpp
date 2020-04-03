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
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <params.hpp>
#include <base.hpp>

namespace cv {
    namespace Matcher {
        class CV_EXPORTS Matcher : protected Base {
        public:
            CV_WRAP Matcher(const cv::Ptr<cv::Matcher::FeatureParams> &_featureParams,
                            const cv::Ptr<cv::flann::IndexParams> &_indexParams,
                            const cv::Ptr<cv::flann::SearchParams> &_searchParams);

            virtual int getVarCount() const CV_OVERRIDE;

            virtual bool empty() const CV_OVERRIDE;

            virtual bool isTrained() const CV_OVERRIDE;

            virtual bool isClassifier() const CV_OVERRIDE;

            virtual bool train(const cv::Ptr<cv::ml::TrainData> &trainData, int flags = 0) CV_OVERRIDE;

            virtual bool train(InputArray samples, int layout, InputArray responses) CV_OVERRIDE;

            virtual bool train(InputArray samples, InputArray template_image, InputArray mask, OutputArray calibrated,
                               float threshold);

            virtual float
            calcError(const cv::Ptr<cv::ml::TrainData> &data, bool test, OutputArray resp) const CV_OVERRIDE;

            virtual float predict(InputArray samples, OutputArray results = noArray(), int flags = 0) const CV_OVERRIDE;

            virtual void write(cv::FileStorage *storage, const char *name) const CV_OVERRIDE;

            virtual void read(cv::FileStorage *storage, CvFileNode *node) CV_OVERRIDE;

            virtual bool create_window();


            static std::string m_window_image, m_window_captured, m_window_clicked, m_window_mask, m_window_matching, m_window_homography;
            bool window_created;
        protected:
            cv::Ptr<cv::BRISK> brisk;
            std::vector<cv::KeyPoint> samples_key_points, template_key_points;
            cv::Mat samples_descriptors, template_descriptors;
            std::vector<cv::Point2f> samples_corners, template_corners;
        };
    }
}

