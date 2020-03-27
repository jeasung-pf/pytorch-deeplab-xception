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

#include <fstream>
#include <memory>
#include <functional>
#include <sys/utsname.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <Classifier.hpp>

static void help(char **argv) {
    std::cout
            << "\nThis program shows how to read in, train on and produce test results for the PASCAL VOC (Visual Object Challenge) data. \n"
            << "It shows how to use detectors, descriptors and recognition methods \n"
               "Using OpenCV version %s\n" << CV_VERSION << "\n"
            << "Call: \n"
            << "Format:\n ./" << argv[0] << " [VOC path] [result directory]  \n"
            << "       or:  \n"
            << " ./" << argv[0]
            << " [VOC path] [result directory] [feature detector] [descriptor extractor] [descriptor matcher] \n"
            << "\n"
            << "Input parameters: \n"
            << "[VOC path]             Path to Pascal VOC data (e.g. /home/my/VOCdevkit/VOC2010). Note: VOC2007-VOC2010 are supported. \n"
            << "[result directory]     Path to result diractory. Following folders will be created in [result directory]: \n"
            << "                         bowImageDescriptors - to store image descriptors, \n"
            << "                         svms - to store trained svms, \n"
            << "                         plots - to store files for plots creating. \n"
            << "[feature detector]     Feature detector name (e.g. SURF, FAST...) - see createFeatureDetector() function in detectors.cpp \n"
            << "                         Currently 12/2010, this is FAST, STAR, SIFT, SURF, MSER, GFTT, HARRIS \n"
            << "[descriptor extractor] Descriptor extractor name (e.g. SURF, SIFT) - see createDescriptorExtractor() function in descriptors.cpp \n"
            << "                         Currently 12/2010, this is SURF, OpponentSIFT, SIFT, OpponentSURF, BRIEF \n"
            << "[descriptor matcher]   Descriptor matcher name (e.g. BruteForce) - see createDescriptorMatcher() function in matchers.cpp \n"
            << "                         Currently 12/2010, this is BruteForce, BruteForce-L1, FlannBased, BruteForce-Hamming, BruteForce-HammingLUT \n"
            << "\n";
}


int main(int argc, char **argv) {

    if (argc != 3 && argc != 5) {
        help(argv);
        return -1;
    }

    struct utsname name;
    if (uname(&name))
        exit(-1);

    std::cout << "Hello! Your computer's OS is" << name.sysname << " " << name.release << std::endl;

    for (int i = 1; i <= 2; ++i) {
        if (argv[i][0] != '/') {
            char *buffer = (char *)malloc(BUFSIZ * sizeof(char));
            if (getcwd(buffer, BUFSIZ) == NULL) {
                CV_Error(CV_StsNullPtr, "Error while retrieving the current working directory.");
            }
            buffer[strlen(buffer) + 1] = 0;
            buffer[strlen(buffer)] = '/';
            argv[i] = strcat(buffer, argv[i]);
            std::cout << argv[i] << std::endl;
        }
    }

    cv::Ptr<cv::Matcher::Classifier> model;
    if (argc == 3) {
        model = new cv::Matcher::Classifier(argv[1], argv[2], "ORB", "lsh");
    } else {
        model = new cv::Matcher::Classifier(argv[1], argv[2], argv[3], argv[4]);
    }
    model->train();

    model.release();
    return 0;
}