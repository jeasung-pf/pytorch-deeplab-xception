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
#include "Classifier.hpp"

namespace cv {
    namespace Matcher {

        extern std::map<std::string, descriptor_algorithm_t> s_map;

        Classifier::Classifier(const std::string _vocPath, const std::string _resPath, const std::string matcher_type)
                : vocPath(_vocPath), resPath(_resPath), matcher_type(matcher_type), Base() {

            std::string vocName;

            makeUsedDirs(resPath);

            cv::FileStorage paramsFS(resPath + "/" + paramsFile, cv::FileStorage::READ);
            if (paramsFS.isOpened()) {
                readUsedParams(paramsFS.root(), vocName, *featureParams, vocabTrainParams, svmTrainParamsExt);
                CV_Assert(vocName == getVocName(_vocPath));
            } else {
                CV_Error(CV_StsBadArg,
                         "Cannot read parameters of this model. Please specify 4 parameters in the class initializer instead of 2 parameters");
            }
        }

        Classifier::Classifier(const std::string _vocPath, const std::string _resPath,
                               const std::string descriptor_type,
                               const std::string matcher_type) : vocPath(_vocPath), resPath(_resPath),
                                                                 matcher_type(matcher_type) {

            std::string vocName;
            makeUsedDirs(resPath);

            cv::FileStorage paramsFS(resPath + "/" + paramsFile, cv::FileStorage::READ);
            vocName = getVocName(vocPath);

            switch (cv::Matcher::s_map[descriptor_type]) {
                case cv::Matcher::MATCHER_DESCRIPTOR_ORB:
                    featureParams = new cv::Matcher::ORBFeatureParams();
                    break;
                case cv::Matcher::MATCHER_DESCRIPTOR_BRISK:
                    featureParams = new cv::Matcher::BRISKFeatureParams();
                    break;
                default:
                    CV_Error(CV_StsBadArg, "Unsupported feature type. Check your implementation");
            }

            // VocabTrainParams and svmTrainParamsExt is set by defaults
            paramsFS.open(resPath + "/" + paramsFile, cv::FileStorage::WRITE);
            if (paramsFS.isOpened()) {
                writeUsedParams(paramsFS, vocName, *featureParams, vocabTrainParams, svmTrainParamsExt);
                paramsFS.release();
            } else {
                CV_Error(CV_StsInternal, "File " + (resPath + "/" + paramsFile) + " can not be opened to write.");
            }
        }

#ifdef HAVE_OPENCV_GPU
        cv::Mat Classifier::trainVocabulary(const std::string &filename, VOCData &vocData,
                                            const cv::Matcher::VocabTrainParams &trainParams,
                                            cv::Ptr<cv::gpu::ORB_GPU> &dextractor) {
#else
        cv::Mat Classifier::trainVocabulary(const std::string &filename, VOCData &vocData,
                                            const cv::Matcher::VocabTrainParams &trainParams,
                                            const cv::Ptr<cv::DescriptorExtractor> &dextractor) {

#endif
            cv::Mat vocabulary;
            if (!cv::Matcher::readVocabulary(filename, vocabulary)) {
#ifdef HAVE_OPENCV_GPU
                // Always support CV_8UC1
                const int elemSize = 1;
#else
                int typeinfo = dextractor->descriptorType();
                if (typeinfo != CV_8UC1)
                    CV_Error(CV_StsAssert, "The type of the descriptor is not CV32FC1, the actual type is " +
                                           std::to_string(typeinfo));
                const int elemSize = CV_ELEM_SIZE(dextractor->descriptorType());
#endif
                const int descByteSize = dextractor->descriptorSize() * elemSize;
                const int bytesInMB = 1048576;
                const int maxDescCount = (trainParams.memoryUse * bytesInMB) / descByteSize;

                std::cout << "Extracting VOC data..." << std::endl;
                std::vector<ObdImage> images;
                std::vector<char> objectPresent;
                vocData.getClassImages(trainParams.trainObjClass, CV_OBD_TRAIN, images, objectPresent);

                std::cout << "Computing descriptors..." << std::endl;
                cv::RNG &rng = cv::theRNG();
                cv::TermCriteria terminate_criterion;
                terminate_criterion.epsilon = FLT_EPSILON;
                cv::BOWKMeansTrainer bowkMeansTrainer(trainParams.vocabSize, terminate_criterion, 3,
                                                      cv::KMEANS_PP_CENTERS);

                while (images.size() > 0) {
                    if (bowkMeansTrainer.descriptorsCount() > maxDescCount) {
#ifdef DEBUG_DESC_PROGRESS
                        std::cout << "Breaking due to full memory ( descriptors count = "
                                  << bowkMeansTrainer.descriptorsCount() << "; descriptor size in bytes = "
                                  << descByteSize << "; all used memory = "
                                  << bowkMeansTrainer.descriptorsCount() * descByteSize << std::endl;
#endif
                        break;
                    }

                    // Randomly pick an image from the dataset which hasn't yet been seen and compute the descriptors
                    // from the image.
                    int randImgIdx = rng((unsigned) images.size());
                    cv::Mat colorImage = cv::imread(images[randImgIdx].path);
                    std::vector<cv::KeyPoint> imageKeypoints;
                    cv::Mat imageDescriptors;
#ifdef HAVE_OPENCV_GPU
                    cv::cvtColor(colorImage, colorImage, CV_RGB2GRAY);

                    cv::gpu::GpuMat colorImage_gpu(colorImage);
                    cv::gpu::GpuMat imageDescriptors_gpu;
                    try {
                        (*dextractor)(colorImage_gpu, cv::gpu::GpuMat(), imageKeypoints, imageDescriptors_gpu);
                    } catch (std::exception e) {
                        std::cout << e.what() << std::endl;
                    }
                    imageDescriptors_gpu.download(imageDescriptors);
#else
                    dextractor->detectAndCompute(colorImage, cv::noArray(), imageKeypoints, imageDescriptors);
#endif
                    // check that there were descriptors calculated for the current image
                    if (!imageDescriptors.empty()) {
                        int descCount = imageDescriptors.rows;
                        // Extract trainParams.descProportion descriptors from the image,
                        // breaking if the 'allDescriptors' matrix becomes full
                        int descsToExtract = static_cast<int >(trainParams.descProportion * static_cast<float >(descCount));
                        // Fill mask of used descriptors
                        std::vector<char> usedMask(descCount, false);
                        std::fill(usedMask.begin(), usedMask.begin() + descsToExtract, true);

                        for (int i = 0; i < descCount; ++i) {
                            int i1 = rng(descCount), i2 = rng(descCount);

                            char tmp = usedMask[i1];
                            usedMask[i1] = usedMask[i2];
                            usedMask[i2] = tmp;
                        }
                        if (imageDescriptors.type() != CV_32FC1) {
                            imageDescriptors.convertTo(imageDescriptors, CV_32FC1);
                        }
                        for (int i = 0; i < descCount; ++i) {
                            if (usedMask[i] && bowkMeansTrainer.descriptorsCount() < maxDescCount) {
                                bowkMeansTrainer.add(imageDescriptors.row(i));
                            }
                        }
                    }
#ifdef DEBUG_DESC_PROGRESS
                    std::cout << images.size() << " images left, " << images[randImgIdx].id << " processed - "
                              << cvRound(
                                      (static_cast<double>(bowkMeansTrainer.descriptorsCount()) /
                                       static_cast<double>(maxDescCount)) *
                                      100.0) << " % memory used"
                              << (imageDescriptors.empty() ? " -> no descriptors extracted, skipping" : "")
                              << std::endl;
#endif
                    // Delete the current element from images so it is not added again
                    images.erase(images.begin() + randImgIdx);
                }

                std::cout << "Maximum allowed descriptor count: " << maxDescCount << ", Actual descriptor count: "
                          << bowkMeansTrainer.descriptorsCount() << std::endl;
                std::cout << "Training vocabulary..." << std::endl;
                vocabulary = bowkMeansTrainer.cluster();

                if (!cv::Matcher::writeVocabulary(filename, vocabulary)) {
                    std::cout << "Error: file " << filename << " can not be opened to write" << std::endl;
                    exit(-1);
                }
            }
            return vocabulary;
        }

        bool Classifier::readBowImageDescriptor(const std::string &file, cv::Mat &bowImageDescriptor) {
            cv::FileStorage fs(file, cv::FileStorage::READ);
            if (fs.isOpened()) {
                fs["imageDescriptor"] >> bowImageDescriptor;
                return true;
            }
            return false;
        }

        bool Classifier::writeBowImageDescriptor(const std::string &file, const cv::Mat &bowImageDescriptor) {
            cv::FileStorage fs(file, cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "imageDescriptor" << bowImageDescriptor;
                return true;
            }
            return false;
        }

/// Load in the bag of words vectors for a set of images, from file if possible
/// \param images
/// \param imageDescriptors
/// \param bowExtractor
/// \param fdetector
/// \param resPath
        void Classifier::calculateImageDescriptors(const std::vector<ObdImage> &images,
                                                   std::vector<cv::Mat> &imageDescriptors,
                                                   cv::Ptr<BOWImgDescriptorExtractorLSH> &bowExtractor,
                                                   const cv::Ptr<cv::FeatureDetector> &fdetector,
                                                   const std::string &resPath) {
            if (bowExtractor->getVocabulary().empty())
                CV_Error(CV_StsAssert, "The vocabulary in the bowExtractor is empty.");
            imageDescriptors.resize(images.size());

            for (size_t i = 0; i < images.size(); i++) {
                std::string filename = resPath + bowImageDescriptorsDir + "/" + images[i].id + ".xml.gz";
                if (readBowImageDescriptor(filename, imageDescriptors[i])) {
#ifdef DEBUG_DESC_PROGRESS
                    std::cout << "Loaded bag of word vector for image " << i + 1 << " of " << images.size() << " ("
                              << images[i].id << ")" << std::endl;
#endif
                } else {
                    cv::Mat colorImage = cv::imread(images[i].path);
#ifdef DEBUG_DESC_PROGRESS
                    std::cout << "Computing descriptors for image " << i + 1 << " of " << images.size() << " ("
                              << images[i].id
                              << ")" << std::flush;
#endif
                    std::vector<cv::KeyPoint> keypoints;
                    fdetector->detect(colorImage, keypoints);
#ifdef DEBUG_DESC_PROGRESS
                    std::cout << " + generating BoW vector" << std::flush;
#endif
//                    if (colorImage.channels() != 1) {
//                        cv::cvtColor(colorImage, colorImage, CV_RGB2GRAY);
//                    }
                    colorImage.convertTo(colorImage, CV_8UC1);
                    bowExtractor->compute(colorImage, keypoints, imageDescriptors[i]);
#ifdef DEBUG_DESC_PROGRESS
                    std::cout << " ...DONE "
                              << static_cast<int>(static_cast<float>(i + 1) / static_cast<float>(images.size()) * 100.0)
                              << " % complete" << std::endl;
#endif
                    if (!imageDescriptors[i].empty()) {
                        if (!writeBowImageDescriptor(filename, imageDescriptors[i])) {
                            std::cout << "Error: file " << filename
                                      << "can not be opened to write bow image descriptor"
                                      << std::endl;
                            std::exit(-1);
                        }
                    }
                }
            }
        }

        void
        Classifier::removeEmptyBowImageDescriptors(std::vector<ObdImage> &images,
                                                   std::vector<cv::Mat> &bowImageDescriptors,
                                                   std::vector<char> &objectPresent) {
            CV_Assert(!images.empty());
            for (int i = (int) images.size() - 1; i >= 0; i--) {
                bool res = bowImageDescriptors[i].empty();
                if (res) {
                    std::cout << "Removing image " << images[i].id << " due to no descriptors..." << std::endl;
                    images.erase(images.begin() + i);
                    bowImageDescriptors.erase(bowImageDescriptors.begin() + i);
                    objectPresent.erase(objectPresent.begin() + i);
                }
            }
        }

        void
        Classifier::removeBowImageDescriptorsByCount(std::vector<ObdImage> &images,
                                                     std::vector<cv::Mat> bowImageDescriptors,
                                                     std::vector<char> objectPresent,
                                                     const cv::Matcher::SVMTrainParamsExt &svmTrainParamsExt,
                                                     int descsToDelete) {
            cv::RNG &rng = cv::theRNG();
            int pos_ex = (int) std::count(objectPresent.begin(), objectPresent.end(), (char) 1);
            int neg_ex = (int) std::count(objectPresent.begin(), objectPresent.end(), (char) 0);

            while (descsToDelete != 0) {
                int randIdx = rng((unsigned) images.size());

                // Prefer positive training examples according to svmTrainParamsExt.targetRatio if required.
                if (objectPresent[randIdx]) {
                    if ((static_cast<float >(pos_ex) / static_cast<float >(neg_ex + pos_ex) <
                         svmTrainParamsExt.targetRatio) &&
                        (neg_ex > 0) && (svmTrainParamsExt.balanceClasses == false)) {
                        continue;
                    } else {
                        pos_ex--;
                    }
                } else {
                    neg_ex--;
                }

                images.erase(images.begin() + randIdx);
                bowImageDescriptors.erase(bowImageDescriptors.begin() + randIdx);
                objectPresent.erase(objectPresent.begin() + randIdx);
                descsToDelete--;
            }
            CV_Assert(bowImageDescriptors.size() == objectPresent.size());
        }

        void
        Classifier::setSVMParams(SvmParams &svmParams, cv::Mat &class_wts_cv, const cv::Mat &responses,
                                 bool balanceClasses) {
            int pos_ex = cv::countNonZero(responses == 1);
            int neg_ex = cv::countNonZero(responses == -1);
            std::cout << pos_ex << " positive training samples; " << neg_ex << " negative training samples"
                      << std::endl;

            svmParams.svmType = cv::ml::SVM::C_SVC;
            svmParams.kernelType = cv::ml::SVM::RBF;
            if (balanceClasses) {
                cv::Mat class_wts(2, 1, CV_32FC1);
                // The first training sample determines the '+1' class internally, even if it is negative,
                // so store whether this is the case so that the class weights can be reversed accordingly.
                bool reversed_classes = (responses.at<float>(0) < 0.f);
                if (reversed_classes == false) {
                    // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
                    class_wts.at<float>(0) = static_cast<float >(pos_ex) / static_cast<float >(pos_ex + neg_ex);
                    // weighting for costs of negative class - 1 (i.e. cost of false negative)
                    class_wts.at<float>(1) = static_cast<float >(neg_ex) / static_cast<float >(pos_ex + neg_ex);
                } else {
                    class_wts.at<float>(0) = static_cast<float >(neg_ex) / static_cast<float >(pos_ex + neg_ex);
                    class_wts.at<float>(1) = static_cast<float >(pos_ex) / static_cast<float >(pos_ex + neg_ex);
                }
                class_wts_cv = class_wts;
                svmParams.classWeights = class_wts_cv;
            }
        }

        void Classifier::setSVMTrainAutoParams(cv::ml::ParamGrid &c_grid, cv::ml::ParamGrid &gamma_grid,
                                               cv::ml::ParamGrid &p_grid, cv::ml::ParamGrid &nu_grid,
                                               cv::ml::ParamGrid &coef_grid, cv::ml::ParamGrid &degree_grid) {

            c_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C);

            gamma_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA);

            p_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P);
            p_grid.logStep = 0;

            nu_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU);
            nu_grid.logStep = 0;

            coef_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF);
            coef_grid.logStep = 0;

            degree_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE);
            degree_grid.logStep = 0;
        }

        void Classifier::trainSVMClassifier(cv::ml::SVM &svm, const cv::Matcher::SVMTrainParamsExt &svmParamsExt,
                                            const std::string &objClassName, VOCData &vocData,
                                            cv::Ptr<BOWImgDescriptorExtractorLSH> &bowExtractor,
                                            const cv::Ptr<cv::DescriptorExtractor> &fdetector,
                                            const std::string &resPath) {

            // first check if a previously trained svm for the current class has been saved to file
            std::string svmFilename = resPath + svmsDir + "/" + objClassName + ".xml.gz";

            cv::FileStorage fs(svmFilename, cv::FileStorage::READ);
            if (fs.isOpened()) {
                std::cout << "***LOADING SVM CLASSIFIER FOR CLASS " << objClassName << " ***" << std::endl;
                svm.load(svmFilename.c_str());
            } else {
                std::cout << "*** TRAINING CLASSIFIER FOR CLASS " << objClassName << " ***" << std::endl;
                std::cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "..." << std::endl;

                // Get classification ground truth for images in the training set
                std::vector<ObdImage> images;
                std::vector<cv::Mat> bowImageDescriptors;
                std::vector<char> objectPresent;
                vocData.getClassImages(objClassName, CV_OBD_TRAIN, images, objectPresent);

                // Compute the bag of words vector for each image in the training set.
                calculateImageDescriptors(images, bowImageDescriptors, bowExtractor, fdetector, resPath);

                // Remove any images for which descriptors could not be calculated
                removeEmptyBowImageDescriptors(images, bowImageDescriptors, objectPresent);

                CV_Assert(svmParamsExt.descPercent > 0.f && svmParamsExt.descPercent <= 1.f);
                if (svmParamsExt.descPercent < 1.f) {
                    int descsToDelete = static_cast<int >(static_cast<float >(images.size()) *
                                                          (1.0 - svmParamsExt.descPercent));

                    std::cout << "Using " << (images.size() - descsToDelete) << " of " << images.size()
                              << " descriptors for training (" << svmParamsExt.descPercent * 100.0 << "%)"
                              << std::endl;
                    removeBowImageDescriptorsByCount(images, bowImageDescriptors, objectPresent, svmParamsExt,
                                                     descsToDelete);
                }
                // Prepare the input matrices for SVM training.
                cv::Mat trainData((int) images.size(), bowExtractor->getVocabulary().rows, CV_32FC1);
                cv::Mat responses((int) images.size(), 1, CV_32SC1);

                // Transfer bag of words vectors and responses across to the training data matrices
                for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++) {
                    // Transfer image descriptor (bag of words vector) to training data matrix
                    cv::Mat submat = trainData.row((int) imageIdx);
                    if (bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize()) {
                        std::cout << "Error: computed bow image descriptor size "
                                  << bowImageDescriptors[imageIdx].cols
                                  << " differs from vocabulary size" << bowExtractor->getVocabulary().cols
                                  << std::endl;
                        std::exit(-1);
                    }
                    bowImageDescriptors[imageIdx].convertTo(bowImageDescriptors[imageIdx], CV_32FC1);
                    bowImageDescriptors[imageIdx].copyTo(submat);
                    // Set response value
                    responses.at<int>((int) imageIdx) = objectPresent[imageIdx] ? 1 : -1;
                }
                cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, responses);

                std::cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << std::endl;
                SvmParams svmParams;
                cv::Mat class_wts_cv;
                setSVMParams(svmParams, class_wts_cv, responses, svmParamsExt.balanceClasses);
                cv::ml::ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
                setSVMTrainAutoParams(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
                svm.trainAuto(data, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
                std::cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << std::endl;

                svm.save(svmFilename.c_str());
                std::cout << "SAVED CLASSIFIER TO FILE" << std::endl;
            }
        }

        void Classifier::computeConfidences(cv::ml::SVM &svm, const std::string &objClassName, VOCData &vocData,
                                            cv::Ptr<BOWImgDescriptorExtractorLSH> &bowExtractor,
                                            const cv::Ptr<cv::DescriptorExtractor> &fdetector,
                                            const std::string &resPath) {

            std::cout << "*** CALCULATING CONFIDENCES FOR CLASS " << objClassName << " ***" << std::endl;
            std::cout << "CALCULATING BOW VECTORS FOR TEST SET OF " << objClassName << "..." << std::endl;
            // Get classification ground truth for images in the test set
            std::vector<ObdImage> images;
            std::vector<cv::Mat> bowImageDescriptors;
            std::vector<char> objectPresent;
            vocData.getClassImages(objClassName, CV_OBD_TEST, images, objectPresent);

            // Compute the bag of words vector for each image in the test set
            calculateImageDescriptors(images, bowImageDescriptors, bowExtractor, fdetector, resPath);
            // Remove any images for which descriptors could not be calculated
            removeEmptyBowImageDescriptors(images, bowImageDescriptors, objectPresent);

            // Use the bag of words vectors to calculate classifier output for each image in test set
            std::cout << "CALCULATING CONFIDENCE SCORES FOR CLASS " << objClassName << "..." << std::endl;
            std::vector<float> confidences(images.size());
            float signMul = 1.f;
            for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++) {
                if (imageIdx == 0) {
                    // In the first iteration, determine the sign of the positive class
                    float classVal = confidences[imageIdx] = svm.predict(bowImageDescriptors[imageIdx]);
                    float scoreVal = confidences[imageIdx] = svm.predict(bowImageDescriptors[imageIdx], cv::noArray(),
                                                                         cv::ml::StatModel::RAW_OUTPUT);
                    signMul = (classVal < 0) == (scoreVal < 0) ? 1.f : -1.f;
                }
                // svm output of decision function
                confidences[imageIdx] = signMul * svm.predict(bowImageDescriptors[imageIdx], cv::noArray(),
                                                              cv::ml::StatModel::RAW_OUTPUT);
            }
            std::cout << "WRITING QUERY RESULTS TO VOC RESULTS FILE FOR CLASS " << objClassName << "..."
                      << std::endl;
            vocData.writeClassifierResultsFile(resPath + plotsDir, objClassName, CV_OBD_TEST, images, confidences,
                                               1,
                                               true);

            std::cout << "DONE - " << objClassName << std::endl;
        }

        void Classifier::computeGnuPlotOutput(const std::string &resPath, const std::string &objClassName,
                                              VOCData &vocData) {
            std::vector<float> precision, recall;
            float ap;

            const std::string resultFile = vocData.getResultsFilename(objClassName, CV_VOC_TASK_CLASSIFICATION,
                                                                      CV_OBD_TEST);
            const std::string plotFile = resultFile.substr(0, resultFile.size() - 4) + ".plt";

            std::cout << "Calculating precision recall curve for class '" << objClassName << "'" << std::endl;
            vocData.calcClassifierPrecRecall(resPath + plotsDir + "/" + resultFile, precision, recall, ap, true);
            std::cout << "Outputting to GNUPlot file..." << std::endl;
            vocData.savePrecRecallToGnuplot(resPath + plotsDir + "/" + plotFile, precision, recall, ap,
                                            objClassName,
                                            CV_VOC_PLOT_PNG);
        }

        bool Classifier::train() {
            // Create detector, descriptor and matcher.
            cv::Ptr<cv::DescriptorExtractor> descExtractor;
            switch (get_param<descriptor_algorithm_t>(*featureParams, "algorithms")) {
                case cv::Matcher::MATCHER_DESCRIPTOR_ORB:
                    descExtractor = cv::ORB::create(get_param<int>(*featureParams, "nfeatures"),
                                                    get_param<float>(*featureParams, "scaleFactor"),
                                                    get_param<int>(*featureParams, "nlevels"),
                                                    get_param<int>(*featureParams, "edgeThreshold"),
                                                    get_param<int>(*featureParams, "firstLevel"),
                                                    get_param<int>(*featureParams, "WTA_K"),
                                                    get_param<int>(*featureParams, "scoreType"),
                                                    get_param<int>(*featureParams, "patchSize"));
                    break;
                case cv::Matcher::MATCHER_DESCRIPTOR_BRISK:
                    descExtractor = cv::BRISK::create(get_param<int>(*featureParams, "thresh"),
                                                      get_param<int>(*featureParams, "octaves"),
                                                      get_param<float>(*featureParams, "patternScale"));
                    break;
                default:
                    CV_Error(CV_StsBadArg, "Unsupported feature descriptor type. Check your implementation.");
            }

#ifdef HAVE_OPENCV_GPU
            cv::Ptr<cv::gpu::ORB_GPU> descExtractor = new cv::gpu::ORB_GPU();
            if (descExtractor.empty()) {
                CV_Error(CV_StsNullPtr, "descExtractor was not created");
            }
#else
            if (descExtractor.empty()) {
                CV_Error(CV_StsNullPtr, "descExtractor was not created");
            }
#endif
            cv::Ptr<cv::DescriptorMatcher> descMatcher = new cv::FlannBasedMatcher(
                    new cv::flann::LshIndexParams(10, 10, 2),
                    new cv::flann::SearchParams());
            cv::Ptr<BOWImgDescriptorExtractorLSH> bowExtractor = new BOWImgDescriptorExtractorLSH(descExtractor, descMatcher);
            if (descMatcher.empty()) {
                CV_Error(CV_StsNullPtr, "descMatcher was not created.");
            }
            // print configuration to screen
            printUsedParams(vocPath, resPath, *featureParams, vocabTrainParams, svmTrainParamsExt);
            // create object to work with VOC
            VOCData vocData(vocPath, false);

            // 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from
#ifdef HAVE_OPENCV_GPU
            cv::Mat vocabulary = trainVocabulary(resPath + "/" + vocabularyFile, vocData, vocabTrainParams,
                                                 descExtractor);
#else
            cv::Mat vocabulary = trainVocabulary(resPath + "/" + vocabularyFile, vocData, vocabTrainParams, descExtractor);
#endif
            vocabulary.convertTo(vocabulary, CV_8UC1);
            bowExtractor->setVocabulary(vocabulary);
            std::cout << "The vocabulary is in type " << vocabulary.type() << std::endl;

            // 2. Train a classifier and run a sample query for each object class
            const std::vector<std::string> &objClasses = vocData.getObjectClasses();
            for (size_t classIdx = 0; classIdx < objClasses.size(); classIdx++) {
                // train a classifier on train dataset
                cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
                trainSVMClassifier(*svm, svmTrainParamsExt, objClasses[classIdx], vocData, bowExtractor,
                                   descExtractor, resPath);
                // now use the classifier over all images on the test dataset and rank according to score order also calculating precision-recall etc.
                computeConfidences(*svm, objClasses[classIdx], vocData, bowExtractor, descExtractor, resPath);
                // calculate precision / recall / ap and use GNUPlot to output to a pdf file
                computeGnuPlotOutput(resPath, objClasses[classIdx], vocData);
            }
            descExtractor.release();
            return true;
        }

        void Classifier::clear() const {

        }

        bool
        Classifier::predict(const cv::Mat &samples, const cv::Mat &sampleMask,
                            cv::Matcher::identification_t &response) const {
            return false;
        }

        int Classifier::getVarCount() const {
            return 0;
        }

        bool Classifier::empty() const {
            return StatModel::empty();
        }

        bool Classifier::isTrained() const {
            return false;
        }

        bool Classifier::isClassifier() const {
            return false;
        }

        bool Classifier::train(const cv::Ptr<cv::ml::TrainData> &trainData, int flags) {
            return StatModel::train(trainData, flags);
        }

        bool Classifier::train(const _InputArray &samples, int layout, const _InputArray &responses) {
            return StatModel::train(samples, layout, responses);
        }

        float Classifier::calcError(const cv::Ptr<cv::ml::TrainData> &data, bool test, const _OutputArray &resp) const {
            return StatModel::calcError(data, test, resp);
        }

        float Classifier::predict(const _InputArray &samples, const _OutputArray &results, int flags) const {
            return 0;
        }

        void Classifier::read(cv::FileStorage *storage, CvFileNode *node) {

        }

        void Classifier::write(cv::FileStorage *storage, const char *name) const {
            Base::write(storage, name);
        }

        BOWImgDescriptorExtractorLSH::BOWImgDescriptorExtractorLSH(const Ptr<DescriptorExtractor> &_dextractor,
                                                                   const Ptr<DescriptorMatcher> &_dmatcher)
                : BOWImgDescriptorExtractor(_dextractor, _dmatcher) {

        }

        void BOWImgDescriptorExtractorLSH::compute(const _InputArray &image, std::vector<KeyPoint> &keypoints,
                                                   const _OutputArray &imgDescriptor,
                                                   std::vector<std::vector<int>> *pointIdxsOfClusters,
                                                   Mat *descriptors) {
            imgDescriptor.release();

            if( keypoints.empty() )
                return;

            // Compute descriptors for the image.
            Mat _descriptors;
            dextractor->compute( image, keypoints, _descriptors );
            std::cout << descriptorType() << std::endl;
            _descriptors.convertTo(_descriptors, CV_8UC1);
            BOWImgDescriptorExtractor::compute( _descriptors, imgDescriptor, pointIdxsOfClusters );

            // Add the descriptors of image keypoints
            if (descriptors) {
                *descriptors = _descriptors.clone();
            }

        }
    }
}








