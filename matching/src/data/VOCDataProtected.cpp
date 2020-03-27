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

#include <data/VOCData.hpp>

std::string getVocName(const std::string &vocPath) {
    size_t found = vocPath.rfind('/');
    if (found == std::string::npos) {
        found = vocPath.rfind('\\');
        if (found == std::string::npos) {
            return vocPath;
        }
    }
    return vocPath.substr(found + 1, vocPath.size() - found);
}

void VOCData::initVoc(const std::string &vocPath, const bool useTestDataset) {
    initVoc2007to2010(vocPath, useTestDataset);
}

void VOCData::initVoc2007to2010(const std::string &vocPath, const bool useTestDataset) {
    // check format of root directory and modify if necessary
    m_vocName = getVocName(vocPath);

    if (m_vocName.compare("VOC2007") && m_vocName.compare("VOC2008") && m_vocName.compare("VOC2009") && m_vocName.compare("VOC2010")) {
        CV_Error(CV_StsAssert, "Unsupported dataset name. Choose one of [VOC2007, VOC2008, VOC2009, VOC2010]");
    }

    m_vocPath = checkFilenamePathsep(vocPath, true);

    if (useTestDataset) {
        m_train_set = "trainval";
        m_test_set = "test";
    } else {
        m_train_set = "train";
        m_test_set = "val";
    }

    // initialize main classification challenge paths
    m_annotation_path = m_vocPath + "Annotations/%s.xml";
    m_image_path = m_vocPath + "JPEGImages/%s.jpg";
    m_imageset_path = m_vocPath + "ImageSets/Main/%s.txt";
    m_class_imageset_path = m_vocPath + "ImageSets/Main/%s_%s.txt";

    //define available object_classes for VOC2010 dataset
    m_object_classes.push_back("aeroplane");
    m_object_classes.push_back("bicycle");
    m_object_classes.push_back("bird");
    m_object_classes.push_back("boat");
    m_object_classes.push_back("bottle");
    m_object_classes.push_back("bus");
    m_object_classes.push_back("car");
    m_object_classes.push_back("cat");
    m_object_classes.push_back("chair");
    m_object_classes.push_back("cow");
    m_object_classes.push_back("diningtable");
    m_object_classes.push_back("dog");
    m_object_classes.push_back("horse");
    m_object_classes.push_back("motorbike");
    m_object_classes.push_back("person");
    m_object_classes.push_back("pottedplant");
    m_object_classes.push_back("sheep");
    m_object_classes.push_back("sofa");
    m_object_classes.push_back("train");
    m_object_classes.push_back("tvmonitor");

    m_min_overlap = 0.5;

    // up until VOC 2010, ap was calculated by sampling p-r curve, not taking complete curve
    m_sampled_ap = ((m_vocName == "VOC2007") || (m_vocName == "VOC2008") || (m_vocName == "VOC2009"));
}

/// Read a VOC classification ground truth text file for a given object class and dataset
/// INPUTS:
/// \param filename : The path of the text file to read
/// OUTPUTS:
/// \param image_codes : VOC image codes extracted from the GT file in the form 20XX_XXXXXX where
///     the first four digits specify the year of the dataset, and the last group specifies a unique ID
/// \param object_present : For each image in the 'image_codes' array, specifies whether the object class described in
///     the loaded GT file is present or not
void VOCData::readClassifierGroundTruth(const std::string &filename, std::vector<std::string> &image_codes,
                                        std::vector<char> &object_present) {
    image_codes.clear();
    object_present.clear();

    std::ifstream gtfile(filename.c_str());
    if (!gtfile.is_open()) {
        std::string err_msg = "could not open VOC ground truth textfile '" + filename + "'.";
        CV_Error(CV_StsError, err_msg.c_str());
    }

    std::string line, image;
    int obj_present = 0;
    while (!gtfile.eof()) {
        std::getline(gtfile, line);
        std::istringstream iss(line);
        iss >> image >> obj_present;

        if (!iss.fail()) {
            image_codes.push_back(image);
            object_present.push_back(obj_present == 1);
        } else {
            if (!gtfile.eof()) {
                CV_Error(CV_StsParseError, "error parsing VOC ground truth textfile.");
            }
        }
    }
    gtfile.close();
}

void VOCData::readClassifierResultsFile(const std::string &input_file, std::vector<std::string> &image_codes,
                                        std::vector<float> &scores) {

    // check if results file exists
    std::ifstream result_file(input_file.c_str());
    if (result_file.is_open()) {
        std::string line, image;
        float score;
        // read in the results file
        while (!result_file.eof()) {
            std::getline(result_file, line);
            std::istringstream iss(line);
            iss >> image >> score;

            if (!iss.fail()) {
                image_codes.push_back(image);
                scores.push_back(score);
            } else {
                if (!result_file.eof()) {
                    CV_Error(CV_StsParseError, "error parsing VOC classifier results file.");
                }
            }
        }
        result_file.close();
    } else {
        std::string err_msg = "could not open classifier results file '" + input_file + "' for reading.";
        CV_Error(CV_StsError, err_msg.c_str());
    }

}

void VOCData::readDetectorResultsFile(const std::string &input_file, std::vector<std::string> &image_codes,
                                      std::vector<std::vector<float> > &scores,
                                      std::vector<std::vector<cv::Rect>> &bounding_boxes) {

}

/// Read a VOC annotation xml file for a given image.
/// INPUTS:
/// \param filename : The path of the xml file to read.
/// \param objects : Array of VocObject describing all object instance present in the given image.
/// \param object_data :
void VOCData::extractVocObjects(const std::string filename, std::vector<ObdObject> &objects,
                                std::vector<VocObjectData> &object_data) {
#ifdef PR_DEBUG
    int block = 1;
    std::cout << "SAMPLE VOC OBJECT EXTRACTION for " << filename << ":" << std::endl;
#endif
    objects.clear();
    object_data.clear();

    std::string contents, object_contents, tag_contents;
    readFileToString(filename, contents);
    // keep on extracting 'object' blocks until no more can be found
    if (extractXMLBlock(contents, "annotation", 0, contents) != -1) {
        int searchpos = 0;
        searchpos = extractXMLBlock(contents, "object", searchpos, object_contents);
        while (searchpos != -1) {
#ifdef PR_DEBUG
            std::cout << "SEARCHPOS:" << searchpos << std::endl;
            std::cout << "start block " << block << " ---------" << std::endl;
            std::cout << object_contents << std::endl;
            std::cout << "end block " << block << " -----------" << std::endl;
            ++block;
#endif
            ObdObject object;
            VocObjectData object_d;

            //object class -------------
            if (extractXMLBlock(object_contents, "name", 0, tag_contents) == -1)
                CV_Error(CV_StsError, "missing <name> tag in object definition of '" + filename + "'");
            object.object_class.swap(tag_contents);

            //object bounding box -------------
            int xmax, xmin, ymax, ymin;

            if (extractXMLBlock(object_contents, "xmax", 0, tag_contents) == -1)
                CV_Error(CV_StsError, "missing <xmax> tag in object definition of '" + filename + "'");
            xmax = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "xmin", 0, tag_contents) == -1)
                CV_Error(CV_StsError, "missing <xmin> tag in object definition of '" + filename + "'");
            xmin = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "ymax", 0, tag_contents) == -1)
                CV_Error(CV_StsError, "missing <ymax> tag in object definition of '" + filename + "'");
            ymax = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "ymin", 0, tag_contents) == -1)
                CV_Error(CV_StsError, "missing <ymin> tag in object definition of '" + filename + "'");
            ymin = stringToInteger(tag_contents);

            object.boundingBox.x = xmin - 1;      //convert to 0-based indexing
            object.boundingBox.width = xmax - xmin;
            object.boundingBox.y = ymin - 1;
            object.boundingBox.height = ymax - ymin;

            CV_Assert(xmin != 0);
            CV_Assert(xmax > xmin);
            CV_Assert(ymin != 0);
            CV_Assert(ymax > ymin);


            //object tags -------------

            if (extractXMLBlock(object_contents, "difficult", 0, tag_contents) != -1) {
                object_d.difficult = (tag_contents == "1");
            } else {
                object_d.difficult = false;
            }

            if (extractXMLBlock(object_contents, "occluded", 0, tag_contents) != -1) {
                object_d.occluded = (tag_contents == "1");
            } else {
                object_d.occluded = false;
            }

            if (extractXMLBlock(object_contents, "truncated", 0, tag_contents) != -1) {
                object_d.truncated = (tag_contents == "1");
            } else {
                object_d.truncated = false;
            }

            if (extractXMLBlock(object_contents, "pose", 0, tag_contents) != -1) {
                if (tag_contents == "Frontal") object_d.pose = CV_VOC_POSE_FRONTAL;
                if (tag_contents == "Rear") object_d.pose = CV_VOC_POSE_REAR;
                if (tag_contents == "Left") object_d.pose = CV_VOC_POSE_LEFT;
                if (tag_contents == "Right") object_d.pose = CV_VOC_POSE_RIGHT;
            }

            //add to array of objects
            objects.push_back(object);
            object_data.push_back(object_d);

            //extract next 'object' block from file if it exists
            searchpos = extractXMLBlock(contents, "object", searchpos, object_contents);
        }
    }
}

/// Converts an image identifier string in the format YYYY_XXXXXX to a single index integer of form XXXXXXYYYY
/// where Y represents a year and returns the image path
/// \param input_str : a string to be modified
/// \return
std::string VOCData::getImagePath(const std::string &input_str) {
    std::string path = m_image_path;
    path.replace(path.find("%s"), 2, input_str);
    return path;
}

/// Extracts the object class and dataset from the filename of a VOC standard results text file, which takes
/// the format 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'
/// \param input_file
/// \param class_name
/// \param dataset_name
void VOCData::extractDataFromResultsFilename(const std::string &input_file, std::string &class_name,
                                             std::string &dataset_name) {
    std::string input_file_std = checkFilenamePathsep(input_file);

    size_t fnamestart = input_file_std.rfind("/");
    size_t fnameend = input_file_std.rfind(".txt");

    if ((fnamestart == input_file_std.npos) || (fnameend == input_file_std.npos)) {
        CV_Error(CV_StsError, "Could not extract filename of results file.");
    }
    fnamestart++;
    if (fnamestart >= fnameend) {
        CV_Error(CV_StsError, "Could not extract filename of results file.");
    }

    // extract dataset and class names, triggering exception if the filename format is not correct
    std::string filename = input_file_std.substr(fnamestart, fnameend - fnamestart);
    size_t datasetstart = filename.find("_");
    datasetstart = filename.find("_", datasetstart + 1);
    size_t classstart = filename.find("_", datasetstart + 1);
    // allow for appended index after a further '_' by discarding this part if it exists
    size_t classend = filename.find("_", classstart + 1);
    if (classend == filename.npos)
        classend = filename.size();

    if ((datasetstart == filename.npos) || (classstart == filename.npos))
        CV_Error(CV_StsError,
                 "Error parsing results filename. Is it in standard format of 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'?");
    ++datasetstart;
    ++classstart;

    if (((datasetstart - classstart) < 1) || ((classend - datasetstart) < 1))
        CV_Error(CV_StsError,
                 "Error parsing results filename. Is it in standard format of 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'?");

    dataset_name = filename.substr(datasetstart, classstart - datasetstart - 1);
    class_name = filename.substr(classstart, classend - classstart);
}

bool VOCData::getClassifierGroundTruthImage(const std::string &obj_class, const std::string &id) {
    // if the classifier ground truth data for all images of the current class has not been loaded yet, load it now
    if (m_classifier_gt_all_ids.empty() || (m_classifier_gt_class != obj_class)) {
        m_classifier_gt_all_ids.clear();
        m_classifier_gt_all_present.clear();
        m_classifier_gt_class = obj_class;

        for (int i = 0; i < 2; ++i) {
            // generate the filename of the classification ground-truth textfile for the object class
            std::string gtFilename = m_class_imageset_path;
            gtFilename.replace(gtFilename.find("%s"), 2, obj_class);

            if (i == 0) {
                gtFilename.replace(gtFilename.find("%s"), 2, m_train_set);
            } else {
                gtFilename.replace(gtFilename.find("%s"), 2, m_test_set);
            }

            // parse the ground truth file, storing in two separate vector for the image code and the ground truth value
            std::vector<std::string> image_codes;
            std::vector<char> object_present;
            readClassifierGroundTruth(gtFilename, image_codes, object_present);

            m_classifier_gt_all_ids.insert(m_classifier_gt_all_ids.end(),
                                           image_codes.begin(),
                                           image_codes.end());
            m_classifier_gt_all_present.insert(m_classifier_gt_all_present.end(),
                                               object_present.begin(),
                                               object_present.end());

            CV_Assert(m_classifier_gt_all_ids.size() == m_classifier_gt_all_present.size());
        }
    }
    // search for the image code
    std::vector<std::string>::iterator it = std::find(m_classifier_gt_all_ids.begin(), m_classifier_gt_all_ids.end(), id);
    if (it != m_classifier_gt_all_ids.end()) {
        // image found, so return corresponding ground truth
        return m_classifier_gt_all_present[std::distance(m_classifier_gt_all_ids.begin(), it)] != 0;
    } else {
        std::string err_msg = "could not find classifier ground truth for image '" + id + "' and class '" + obj_class + "'";
        CV_Error(CV_StsError,err_msg.c_str());
    }

    return true;
}