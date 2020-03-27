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

#include "../include/Bootstrap.hpp"
#include "../include/defines.hpp"

#include <cfloat>
#include <cmath>

cv::Matcher::Bootstrap::Bootstrap(const cv::Matcher::FeatureParams &_featureParams,
		const cv::flann::IndexParams &_indexParams,
		const cv::flann::SearchParams &_searchParams): Base(_featureParams, _indexParams, _searchParams) {

}

void cv::Matcher::Bootstrap::clear() {
	Base::clear();
}

bool cv::Matcher::Bootstrap::predict(const cv::Mat &sample, const cv::Mat &sampleMask,
                                     cv::Matcher::identification_t &response, int k) const {

	std::vector<cv::KeyPoint> sampleKeyPoints;
	cv::Mat sampleDescriptor;
	vector<float> similarity((int)cv::Matcher::identification_t::COUNT, 0.0);

	Base::extract(sample, sampleMask, sampleKeyPoints, sampleDescriptor);

	// Match the given image with the list of template images
	for (int i = 0; i < templateKeyPoints.size(); i++) {
		
		std::vector<std::vector<cv::DMatch>> matches;
		std::vector<std::vector<float>> cost(sampleDescriptor.rows, std::vector<float>(this->templateDescriptors[i].rows, FLT_MAX));
		std::vector<cv::Vec2f> sampleSignature, trainSignature;

		this->matcher->knnMatch(sampleDescriptor, this->templateDescriptors[i], matches, k);

		for (const auto _ : matches) {
		    for (auto elem : _) {
                cost[elem.queryIdx][elem.trainIdx] = elem.distance;
            }
		}

		int j = 0;
		for (const auto elem : this->templateKeyPoints[i]) {
			trainSignature.push_back(cv::Vec2f((float)(int)pow(elem.size, 2.0), (float)j));
			j++;
		}

		j = 0;
		for (auto elem : sampleKeyPoints) {
			sampleSignature.push_back(cv::Vec2f((float)(int)pow(elem.size, 2.0), (float)j));
			j++;
		}
		
		try {
			similarity[i] = cv::EMD(sampleSignature, trainSignature, CV_DIST_USER, cv::Mat(cost), 0, cv::noArray());
		} catch(std::exception e) {
			throw cv::Matcher::MatcherException(std::string("Error while computing Earth Mover's Distance. Consult your implementation or data.\n"));
			return false;
		}
	}
	// compare two image using Earth Mover's Distance
	int max_index = std::max_element(similarity.begin(), similarity.end()) - similarity.begin();

	switch(max_index) {
		case 0:
			response = cv::Matcher::identification_t::ID_REGISTRATION_CARD;
			break;
		case 1:
			response = cv::Matcher::identification_t::ID_DRIVERS_LICENSE;
			break;
		default:
			throw cv::Matcher::MatcherException(std::string("The kind of ID card is not in the list. Please add them and try again.\n"));
	}
	
	return true;
}

cv::Matcher::Bootstrap::~Bootstrap() {

}

void cv::Matcher::Bootstrap::write(CvFileStorage *storage, const char *name) const {
    Base::write(storage, name);
}

void cv::Matcher::Bootstrap::read(CvFileStorage *storage, CvFileNode *node) {
    Base::read(storage, node);
}

