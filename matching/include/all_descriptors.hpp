/************************************************************************
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

#ifndef OPENCV_MATCHER_ALL_DESCRIPTORS_HPP_
#define OPENCV_MATCHER_ALL_DESCRIPTORS_HPP_

#include "./defines.hpp"
#include "./params.hpp"

#include <opencv2/features2d/features2d.hpp>

namespace cv { namespace Matcher {

	struct feature_creator
	{
		static cv::Ptr<cv::Feature2D> create(const cv::Matcher::FeatureParams &params)
		{
			descriptor_algorithm_t descriptor_type = get_param<descriptor_algorithm_t>(params, "algorithms");

			cv::Ptr<cv::Feature2D> feature_descriptor;

			switch (descriptor_type) {
				case MATCHER_DESCRIPTOR_ORB:
				    feature_descriptor = cv::ORB::create(get_param<int>(params, "nfeatures"),
                                                         get_param<float>(params, "scaleFactor"),
                                                         get_param<int>(params, "nlevels"),
                                                         get_param<int>(params, "edgeThreshold"),
                                                         get_param<int>(params, "firstLevel"),
                                                         get_param<int>(params, "WTA_K"),
                                                         get_param<int>(params, "scoreType"),
                                                         get_param<int>(params, "patchSize"));
					break;
				case MATCHER_DESCRIPTOR_BRISK:
				    feature_descriptor = cv::BRISK::create(get_param<int>(params, "threshold"),
                                                           get_param<int>(params, "octaves"),
                                                           get_param<int>(params, "patternSCale"));
					break;
			}
			return feature_descriptor;
		}
	};
}}

#endif
