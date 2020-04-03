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
                        float _filterThreshold) : featureParams(_featureParams), indexParams(_indexParams), searchParams(_searchParams) {

    matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
}

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





