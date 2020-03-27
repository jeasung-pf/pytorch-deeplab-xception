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

#ifndef OPENCV_CLASSIFIER_HPP_
#define OPENCV_CLASSIFIER_HPP_

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "base.hpp"
#include "params.hpp"
#include "defines.hpp"
#include "data/VOCData.hpp"

#ifdef HAVE_OPENCV_OCL
#define _OCL_SVM_ 1
#include <opencv2/ocl/ocl.hpp>
#endif

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif

namespace cv {
    namespace Matcher {

        // SVM training parameters
        struct SvmParams
        {
            int         svmType;
            int         kernelType;
            double      gamma;
            double      coef0;
            double      degree;
            double      C;
            double      nu;
            double      p;
            cv::Mat         classWeights;
            cv::TermCriteria termCrit;

            SvmParams()
            {
                svmType = cv::ml::SVM::C_SVC;
                kernelType = cv::ml::SVM::RBF;
                degree = 0;
                gamma = 1;
                coef0 = 0;
                C = 1;
                nu = 0;
                p = 0;
                termCrit = cv::TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
            }

            SvmParams( int _svmType, int _kernelType,
                       double _degree, double _gamma, double _coef0,
                       double _Con, double _nu, double _p,
                       const cv::Mat& _classWeights, cv::TermCriteria _termCrit )
            {
                svmType = _svmType;
                kernelType = _kernelType;
                degree = _degree;
                gamma = _gamma;
                coef0 = _coef0;
                C = _Con;
                nu = _nu;
                p = _p;
                classWeights = _classWeights;
                termCrit = _termCrit;
            }

        };

        class BOWImgDescriptorExtractorLSH : public cv::BOWImgDescriptorExtractor {
        public:
            BOWImgDescriptorExtractorLSH(const Ptr<DescriptorExtractor>& _dextractor,
                                         const Ptr<DescriptorMatcher>& _dmatcher);

            void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray imgDescriptor,
                          std::vector<std::vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 );
        };

        class CV_EXPORTS_W Classifier : protected Base {
        public:
            Classifier(const std::string _vocPath, const std::string _resPath, const std::string matcher_type);

            Classifier(const std::string _vocPath, const std::string _resPath,
                       const std::string descriptor_type = "ORB", const std::string matcher_type = "LSH");

            virtual ~Classifier() {

            }

            virtual void clear() const;

            virtual int getVarCount() const CV_OVERRIDE;

            virtual bool empty() const CV_OVERRIDE;

            virtual bool isTrained() const CV_OVERRIDE;

            virtual bool isClassifier() const CV_OVERRIDE;

            virtual bool train( const cv::Ptr<cv::ml::TrainData>& trainData, int flags=0 ) CV_OVERRIDE;

            virtual bool train( InputArray samples, int layout, InputArray responses ) CV_OVERRIDE;

            virtual float calcError( const cv::Ptr<cv::ml::TrainData>& data, bool test, OutputArray resp ) const CV_OVERRIDE;

            virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const CV_OVERRIDE;


            virtual bool train();

            virtual bool
            predict(const cv::Mat &samples, const cv::Mat &sampleMask, cv::Matcher::identification_t &response) const;


            virtual void write(cv::FileStorage *storage, const char *name) const CV_OVERRIDE;

            virtual void read(cv::FileStorage *storage, CvFileNode *node) CV_OVERRIDE;

        protected:


#ifdef HAVE_OPENCV_GPU
            cv::Mat trainVocabulary(const std::string &filename, VOCData &vocData,
                                    const cv::Matcher::VocabTrainParams &trainParams,
                                    cv::Ptr<cv::gpu::ORB_GPU> &dextractor);
#else

            cv::Mat trainVocabulary(const std::string &filename, VOCData &vocData,
                                    const cv::Matcher::VocabTrainParams &trainParams,
                                    const cv::Ptr<cv::DescriptorExtractor> &dextractor);

#endif

            bool readBowImageDescriptor(const std::string &file, cv::Mat &bowImageDescriptor);

            bool writeBowImageDescriptor(const std::string &file, const cv::Mat &bowImageDescriptor);

            void calculateImageDescriptors(const std::vector<ObdImage> &images, std::vector<cv::Mat> &imageDescriptors,
                                           cv::Ptr<BOWImgDescriptorExtractorLSH> &bowExtractor,
                                           const cv::Ptr<cv::FeatureDetector> &fdetector, const std::string &resPath);

            void
            removeEmptyBowImageDescriptors(std::vector<ObdImage> &images, std::vector<cv::Mat> &bowImageDescriptors,
                                           std::vector<char> &objectPresent);

            void
            removeBowImageDescriptorsByCount(std::vector<ObdImage> &images, std::vector<cv::Mat> bowImageDescriptors,
                                             std::vector<char> objectPresent,
                                             const cv::Matcher::SVMTrainParamsExt &svmTrainParamsExt,
                                             int descsToDelete);

            void
            setSVMParams(SvmParams &svmParams, cv::Mat &class_wts_cv, const cv::Mat &responses,
                         bool balanceClasses);

            void
            setSVMTrainAutoParams(cv::ml::ParamGrid &c_grid, cv::ml::ParamGrid &gamma_grid, cv::ml::ParamGrid &p_grid,
                                  cv::ml::ParamGrid &nu_grid, cv::ml::ParamGrid &coef_grid,
                                  cv::ml::ParamGrid &degree_grid);

            void
            trainSVMClassifier(cv::ml::SVM &svm, const cv::Matcher::SVMTrainParamsExt &svmParamsExt,
                               const std::string &objClassName,
                               VOCData &vocData, cv::Ptr<BOWImgDescriptorExtractorLSH> &bowExtractor,
                               const cv::Ptr<cv::DescriptorExtractor> &fdetector, const std::string &resPath);

            void computeConfidences(cv::ml::SVM &svm, const std::string &objClassName, VOCData &vocData,
                                    cv::Ptr<BOWImgDescriptorExtractorLSH> &bowExtractor,
                                    const cv::Ptr<cv::DescriptorExtractor> &fdetector, const std::string &resPath);

            void computeGnuPlotOutput(const std::string &resPath, const std::string &objClassName, VOCData &vocData);

        private:
            std::string vocPath;
            std::string resPath;
            std::string matcher_type;
            cv::Matcher::VocabTrainParams vocabTrainParams;
            cv::Matcher::SVMTrainParamsExt svmTrainParamsExt;
        };
    }
}

#endif
