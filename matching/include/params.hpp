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


#ifndef OPENCV_FEATURE_PARAMS_H_
#define OPENCV_FEATURE_PARAMS_H_

#include <iostream>
#include <map>

#include <opencv2/flann/flann.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "general.hpp"
#include "defines.hpp"
#include "data/VOCData.hpp"

namespace cv {
    namespace Matcher {
        typedef std::map<std::string, cvflann::any> FeatureParams;

        template<typename T>
        extern T get_param(const FeatureParams &params, std::string name, const T &default_value);

        template<typename T>
        extern T get_param(const FeatureParams &params, std::string name);

        extern inline void print_params(const FeatureParams &params, std::ostream &stream);

        extern inline void print_params(const FeatureParams &params);


        struct ORBFeatureParams : public FeatureParams {
            ORBFeatureParams(
                    int nfeatures = 500,
                    float scaleFactor = 1.2f,
                    int nlevels = 8,
                    int edgeThreshold = 31,
                    int firstLevel = 0,
                    int WTA_K = 2,
                    int scoreType = cv::ORB::HARRIS_SCORE,
                    int patchSize = 31) {

                (*this)["algorithms"] = cv::Matcher::MATCHER_DESCRIPTOR_ORB;
                (*this)["nfeatures"] = nfeatures;
                (*this)["scaleFactor"] = scaleFactor;
                (*this)["nlevels"] = nlevels;
                (*this)["edgeThreshold"] = edgeThreshold;
                (*this)["firstLevel"] = firstLevel;
                (*this)["WTA_K"] = 2;
                (*this)["scoreType"] = scoreType;
                (*this)["patchSize"] = patchSize;
            }

            virtual void read(const cv::FileNode &fn) {
                (*this)["algorithms"] = (cv::Matcher::descriptor_algorithm_t) (int) fn["algorithms"];
                (*this)["nfeatures"] = (int) fn["nfeatures"];
                (*this)["scaleFactor"] = (float) fn["scaleFactor"];
                (*this)["nlevels"] = (int) fn["nlevels"];
                (*this)["edgeThreshold"] = (int) fn["edgeThreshold"];
                (*this)["firstLevel"] = (int) fn["firstLevel"];
                (*this)["WTA_K"] = (int) fn["WTA_K"];
                (*this)["scoreType"] = (int) fn["scoreType"];
                (*this)["patchSize"] = (int) fn["patchSize"];
            }

            virtual void write(cv::FileStorage &fn) {
                fn << "algorithms" << get_param<cv::Matcher::descriptor_algorithm_t>(*this, "algorithms");
                fn << "nfeatures" << get_param<int>(*this, "nfeatures");
                fn << "scaleFactor" << get_param<float>(*this, "scaleFactor");
                fn << "nlevels" << get_param<int>(*this, "nlevels");
                fn << "edgeThreshold" << get_param<int>(*this, "edgeThreshold");
                fn << "firstLevel" << get_param<int>(*this, "firstLevel");
                fn << "WTA_K" << get_param<int>(*this, "WTA_K");
                fn << "scoreType" << get_param<int>(*this, "scoreType");
                fn << "patchSize" << get_param<int>(*this, "patchSize");
            }

            virtual void print() {
                std::cout << "algorithms: " << get_param<cv::Matcher::descriptor_algorithm_t>(*this, "algorithms")
                          << std::endl;
                std::cout << "nfeatures: " << get_param<int>(*this, "nfeatures") << std::endl;
                std::cout << "scaleFactor: " << get_param<float>(*this, "scaleFactor") << std::endl;
                std::cout << "nlevels: " << get_param<int>(*this, "nlevels") << std::endl;
                std::cout << "edgeThreshold: " << get_param<int>(*this, "edgeThreshold") << std::endl;
                std::cout << "firstLevel: " << get_param<int>(*this, "firstLevel") << std::endl;
                std::cout << "WTA_K: " << get_param<int>(*this, "WTA_K") << std::endl;
                std::cout << "scoreType: " << get_param<int>(*this, "scoreType") << std::endl;
                std::cout << "patchSize: " << get_param<int>(*this, "patchSize") << std::endl;
            }
        };

        struct BRISKFeatureParams : public FeatureParams {
            BRISKFeatureParams(
                    int threshold = 30,
                    int octaves = 3,
                    float patternScale = 1.0f
            ) {

                (*this)["algorithms"] = cv::Matcher::MATCHER_DESCRIPTOR_BRISK;
                (*this)["threshold"] = threshold;
                (*this)["octaves"] = octaves;
                (*this)["patternScale"] = patternScale;
            }

            virtual void read(const cv::FileNode &fn) {
                (*this)["algorithms"] = (cv::Matcher::descriptor_algorithm_t) (int) fn["algorithms"];
                (*this)["threshold"] = (int) fn["threshold"];
                (*this)["octaves"] = (int) fn["octaves"];
                (*this)["patternScale"] = (float) fn["patternScale"];
            }

            virtual void write(cv::FileStorage &fs) {
                fs << "algorithms" << get_param<cv::Matcher::descriptor_algorithm_t>(*this, "algorithms");
                fs << "threshold" << get_param<int>(*this, "threshold");
                fs << "octaves" << get_param<int>(*this, "octaves");
                fs << "patternScale" << get_param<float>(*this, "patternScale");
            }

            virtual void print() const {
                std::cout << "algorithms: " << get_param<cv::Matcher::descriptor_algorithm_t>(*this, "algorithms")
                          << std::endl;
                std::cout << "threshold: " << get_param<int>(*this, "threshold") << std::endl;
                std::cout << "octaves: " << get_param<int>(*this, "octaves") << std::endl;
                std::cout << "patternScale: " << get_param<float>(*this, "patternScale") << std::endl;
            }
        };

        struct VocabTrainParams {

            VocabTrainParams() : trainObjClass("chair"), vocabSize(1000), memoryUse(200), descProportion(0.3f) {

            }

            VocabTrainParams(const std::string _trainObjClass, size_t _vocabSize, size_t _memoryUse,
                             float _descProportion) : trainObjClass(_trainObjClass), vocabSize((int) _vocabSize),
                                                      memoryUse((int) _memoryUse), descProportion(_descProportion) {

            }

            void read(const cv::FileNode &fn) {
                fn["trainObjClass"] >> trainObjClass;
                fn["vocabSize"] >> vocabSize;
                fn["memoryUse"] >> memoryUse;
                fn["descProportion"] >> descProportion;
            }

            void write(cv::FileStorage &fs) const {
                fs << "trainObjClass" << trainObjClass;
                fs << "vocabSize" << vocabSize;
                fs << "memoryUse" << memoryUse;
                fs << "descProportion" << descProportion;
            }

            void print() const {
                std::cout << "trainObjClass: " << trainObjClass << std::endl;
                std::cout << "vocabSize: " << vocabSize << std::endl;
                std::cout << "memoryUse: " << memoryUse << std::endl;
                std::cout << "descProportion: " << descProportion << std::endl;
            }

            // Object class used for training visual vocabulary.
            // It should not matter which object class is specified here. Visual vocab will still be the same.
            std::string trainObjClass;
            // number of visual words in vocabulary to train.
            int vocabSize;
            // Memory to preallocate (in MB) when training vocab.
            // Change this depending on the size of the dataset / available memory
            int memoryUse;
            // Specifies the number of descriptors to use from each image as a proportion of the total num descs
            float descProportion;
        };

        struct SVMTrainParamsExt {

            SVMTrainParamsExt() : descPercent(0.5f), targetRatio(0.4f), balanceClasses(true) {

            }

            SVMTrainParamsExt(float _descPercent, float _targetRatio, bool _balanceClasses) : descPercent(_descPercent),
                                                                                              targetRatio(_targetRatio),
                                                                                              balanceClasses(
                                                                                                      _balanceClasses) {

            }

            void read(const cv::FileNode &fn) {
                fn["descPercent"] >> descPercent;
                fn["targetRatio"] >> targetRatio;
                fn["balanceClasses"] >> balanceClasses;
            }

            void write(cv::FileStorage &fs) const {
                fs << "descPercent" << descPercent;
                fs << "targetRatio" << targetRatio;
                fs << "balanceClasses" << balanceClasses;
            }

            void print() const {
                std::cout << "descPercent: " << descPercent << std::endl;
                std::cout << "targetRatio: " << targetRatio << std::endl;
                std::cout << "balanceClasses: " << balanceClasses << std::endl;
            }

            // Percentage of extracted descriptors to use for training.
            float descPercent;
            // Try to get this ratio of positive to negative samples (minimum).
            float targetRatio;
            // Balance class weights by number of samples in each (if true cSvmTrainTargetRatio is ignored).
            bool balanceClasses;
        };

        extern bool readVocabulary(const std::string &filename, cv::Mat &vocabulary);

        extern bool writeVocabulary(const std::string &filename, const cv::Mat &vocabulary);

        extern void readUsedParams(const cv::FileNode &fn, std::string &vocName,
                                   cv::Matcher::FeatureParams &featureParams,
                                   cv::Matcher::VocabTrainParams &vocabTrainParams,
                                   cv::Matcher::SVMTrainParamsExt &svmTrainParamsExt);

        extern void
        writeUsedParams(cv::FileStorage &fs, const std::string &vocName,
                        const cv::Matcher::FeatureParams &featureParams,
                        const cv::Matcher::VocabTrainParams &vocabTrainParams,
                        const cv::Matcher::SVMTrainParamsExt &svmTrainParamsExt);

        extern void
        printUsedParams(const std::string &vocPath, const std::string &resDir,
                        const cv::Matcher::FeatureParams &featureParams,
                        const cv::Matcher::VocabTrainParams &vocabTrainParams,
                        const cv::Matcher::SVMTrainParamsExt &svmTrainParamsExt);
    }
}

#endif
