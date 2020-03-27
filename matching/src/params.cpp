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

#include <params.hpp>

namespace cv {
    namespace Matcher {

        template<typename T>
        T get_param(const FeatureParams &params, std::string name, const T &default_value) {
            FeatureParams::const_iterator it = params.find(name);

            if (it != params.end()) {
                return it->second.cast<T>();
            } else {
                return default_value;
            }
        }

        template<typename T>
        T get_param(const FeatureParams &params, std::string name) {

            FeatureParams::const_iterator it = params.find(name);

            if (it != params.end()) {
                return it->second.cast<T>();
            } else {
                throw MatcherException(
                        std::string("Missing parameter '") + name + std::string("' in the parameters given"));
            }
        }

        inline void print_params(const FeatureParams &params, std::ostream &stream) {
            FeatureParams::const_iterator it;

            for (it = params.begin(); it != params.end(); it++) {
                stream << it->first << " : " << it->second << std::endl;
            }
        }

        inline void print_params(const FeatureParams &params) {
            print_params(params, std::cout);
        }

        void readUsedParams(const cv::FileNode &fn, std::string &vocName, FeatureParams &featureParams,
                            VocabTrainParams &vocabTrainParams, SVMTrainParamsExt &svmTrainParamsExt) {
            fn["vocName"] >> vocName;

            cv::FileNode currFn = fn;

            currFn = fn["featureParams"];
            switch (fn["algorithms"]) {
                case MATCHER_DESCRIPTOR_ORB: {
                    ORBFeatureParams oParams;
                    oParams = static_cast<ORBFeatureParams &>(featureParams);
                    oParams.read(currFn);
                    break;
                }
                case MATCHER_DESCRIPTOR_BRISK: {
                    BRISKFeatureParams bParams;
                    bParams = static_cast<BRISKFeatureParams &>(featureParams);
                    bParams.read(currFn);
                    break;
                }
                default:
                    CV_Error(CV_StsBadArg, "Unsupported feature descriptor type. Check your implementation.");
            }

            currFn = fn["vocabTrainParams"];
            vocabTrainParams.read(fn);

            currFn = fn["svmTrainParamsExt"];
            svmTrainParamsExt.read(currFn);
        }

        void writeUsedParams(cv::FileStorage &fs, const std::string &vocName, const FeatureParams &featureParams,
                             const VocabTrainParams &vocabTrainParams, const SVMTrainParamsExt &svmTrainParamsExt) {
            fs << "vocName" << vocName;

            fs << "featureParams" << "{";
            switch (const_cast<FeatureParams &>(featureParams)["algorithms"].cast<descriptor_algorithm_t>()) {
                case MATCHER_DESCRIPTOR_ORB: {
                    ORBFeatureParams oParams = static_cast<ORBFeatureParams &>(const_cast<FeatureParams &>(featureParams));
                    oParams.write(fs);
                    break;
                }
                case MATCHER_DESCRIPTOR_BRISK: {
                    BRISKFeatureParams bParams = static_cast<BRISKFeatureParams &>(const_cast<FeatureParams &>(featureParams));
                    bParams.write(fs);
                    break;
                }
                default:
                    CV_Error(CV_StsBadArg, "Unsupported feature descriptor type. Check your implementation.");
            }
            fs << "}";

            fs << "vocabTrainParams" << "{";
            vocabTrainParams.write(fs);
            fs << "}";

            fs << "svmTrainParamsExt" << "{";
            svmTrainParamsExt.write(fs);
            fs << "}";
        }

        void printUsedParams(const std::string &vocPath, const std::string &resDir, const FeatureParams &featureParams,
                             const VocabTrainParams &vocabTrainParams, const SVMTrainParamsExt &svmTrainParamsExt) {
            std::cout << "CURRENT CONFIGURATION" << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
            std::cout << "vocPath: " << vocPath << std::endl;
            std::cout << "resDir: " << resDir << std::endl;

            switch (const_cast<FeatureParams &>(featureParams)["algorithms"].cast<descriptor_algorithm_t>()) {
                case cv::Matcher::MATCHER_DESCRIPTOR_ORB: {
                    ORBFeatureParams oParams = static_cast<ORBFeatureParams &>(const_cast<FeatureParams &>(featureParams));
                    oParams.print();
                    break;
                }
                case cv::Matcher::MATCHER_DESCRIPTOR_BRISK: {
                    BRISKFeatureParams bParams = static_cast<BRISKFeatureParams &>(const_cast<FeatureParams &>(featureParams));
                    bParams.print();
                    break;
                }
                default:
                    CV_Error(CV_StsBadArg, "Unsupported feature descriptor type. Check your implementation.");
            }

            std::cout << std::endl;
            vocabTrainParams.print();
            std::cout << std::endl;
            svmTrainParamsExt.print();
            std::cout << "----------------------------------------------------------------" << std::endl << std::endl;
        }

        bool readVocabulary(const std::string &filename, cv::Mat &vocabulary) {
            std::cout << "Reading vocabulary..." << std::endl;
            cv::FileStorage fs(filename, cv::FileStorage::READ);
            if (fs.isOpened()) {
                fs["vocabulary"] >> vocabulary;
                std::cout << "done" << std::endl;
                return true;
            }
            return false;
        }

        bool writeVocabulary(const std::string &filename, const cv::Mat &vocabulary) {
            std::cout << "Saving vocabulary to " << filename << std::endl;
            cv::FileStorage fs(filename, cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "vocabulary" << vocabulary;
                return true;
            }
            return false;
        }

    }
}