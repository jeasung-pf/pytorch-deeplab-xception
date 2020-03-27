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

void makeDir(const std::string &dir) {
    if (dir[0] != '/') {
        char buffer[BUFSIZ];
        if ((getcwd(buffer, sizeof(buffer)) == NULL)) {
            CV_Error(CV_StsNullPtr, "Error while retrieving the current working directory.");
        }
        std::string cwd(buffer);
        cwd.append("/" + dir);
        mkdir(cwd.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    } else {
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

void makeUsedDirs(const std::string &rootPath) {
    makeDir(rootPath);
    makeDir(rootPath + bowImageDescriptorsDir);
    makeDir(rootPath + svmsDir);
    makeDir(rootPath + plotsDir);
}

/// Return the classification ground truth data for all images of a given VOC object class
/// INPUTS:
/// \param obj_class : The VOC object class identifier string
/// \param dataset : Specifies whether to extract images from the training or test set
/// OUTPUTS:
/// \param images : An array of ObdImage containing info of all images extracted from the ground truth file
/// \param object_present : An array of bools specifying whether the object defined by 'obj_class' is present in each
///     image or not.
void VOCData::getClassImages(const std::string &obj_class,
                             const ObdDatasetType dataset, std::vector<ObdImage> &images,
                             std::vector<char> &object_present) {

    std::string dataset_str;
    // generate the filename of the classification ground-truth textfile for the object class
    if (dataset == CV_OBD_TRAIN) {
        dataset_str = m_train_set;
    } else {
        dataset_str = m_test_set;
    }
    getClassImages_impl(obj_class, dataset_str, images, object_present);
}

/// Return ground truth data for the objects present in an image with a given UID
/// INPUTS:
/// \param id : VOC Dataset unique identifier (string code in form YYYY_XXXXXX where YYYY is the year)
/// OUTPUTS:
/// \param obj_class : Specifies the object class to use to resolve 'ground_truth'
/// \param objects : Contains the extended object info (bounding box etc.) for each object in the image
/// \param object_data : Contains VOC-specific extended object info (marked difficult etc.)
/// \param ground_truth : Specifies whether there are any difficult/non-difficult instances of
///     the current object class within the image
/// \return ObdImage containing path and other details of image file with given code
ObdImage VOCData::getObjects(const std::string &id, std::vector<ObdObject> &objects) {
    std::vector<VocObjectData> object_data;
    ObdImage image = getObjects(id, objects, object_data);

    return image;
}

ObdImage
VOCData::getObjects(const std::string &id, std::vector<ObdObject> &objects, std::vector<VocObjectData> &object_data) {
    // first generate the filename of the annotation file
    std::string annotationFilename = m_annotation_path;
    annotationFilename.replace(annotationFilename.find("%s"), 2, id);
    // extract objects contained in the current image from the xml
    extractVocObjects(annotationFilename, objects, object_data);
    // generate image path from extracted string code
    std::string path = getImagePath(id);

    ObdImage image(id, path);
    return image;
}

ObdImage VOCData::getObjects(const std::string &obj_class, const std::string &id, std::vector<ObdObject> &objects,
                             std::vector<VocObjectData> &object_data, VocGT &ground_truth) {
    // extract object data (except for ground truth flag)
    ObdImage image = getObjects(id, objects, object_data);
    // pregenerate a flag to indicate whether the current class is present or not in the image
    ground_truth = CV_VOC_GT_NONE;
    for (size_t j = 0; j < objects.size(); j++) {
        if (objects[j].object_class == obj_class) {
            if (object_data[j].difficult == false) {
                // if at least one non difficult example is present, this flag is always set to CV_VOC_GT_PRESENT
                ground_truth = CV_VOC_GT_PRESENT;
                break;
            } else {
                ground_truth = CV_VOC_GT_DIFFICULT;
            }
        }
    }

    return image;
}

void VOCData::getClassImages_impl(const std::string &obj_class, const std::string &dataset_str,
                                  std::vector<ObdImage> &images, std::vector<char> &object_present) {
    // generates the filename of the classification ground-truth textfile for the object class
    std::string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"), 2, obj_class);
    gtFilename.replace(gtFilename.find("%s"), 2, dataset_str);

    // parse the ground truth file, storing in two separate vectors for the image code and the ground truth value
    std::vector<std::string> image_codes;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    // prepare output arrays
    images.clear();

    convertImageCodesToObdImages(image_codes, images);
}

/// Return the ground truth data for the presence/absence of a given object class in an arbitrary array of image.
/// INPUTS:
/// \param obj_class : The VOC object class identifier string
/// \param images : An array of ObdImage OR strings containing the images for which ground truth will be computed
/// OUTPUTS:
/// \param ground_truth : An output array indicating the presence/absence of obj_class within each image
void VOCData::getClassifierGroundTruth(const std::string &obj_class, const std::vector<ObdImage> &images,
                                       std::vector<char> &ground_truth) {
    std::vector<char>(images.size()).swap(ground_truth);

    std::vector<ObdObject> objects;
    std::vector<VocObjectData> object_data;
    std::vector<char>::iterator gt_it = ground_truth.begin();

    for (std::vector<ObdImage>::const_iterator it = images.begin(); it != images.end(); it++, gt_it++) {
        (*gt_it) = (getClassifierGroundTruthImage(obj_class, it->id));
    }
}

void VOCData::getClassifierGroundTruth(const std::string &obj_class, const std::vector<std::string> &images,
                                       std::vector<char> &ground_truth) {
    std::vector<char>(images.size()).swap(ground_truth);

    std::vector<ObdObject> objects;
    std::vector<VocObjectData> object_data;
    std::vector<char>::iterator gt_it = ground_truth.begin();

    for (std::vector<std::string>::const_iterator it = images.begin(); it != images.end(); it++, gt_it++) {
        (*gt_it) = (getClassifierGroundTruthImage(obj_class, (*it)));
    }
}

/// Write VOC-compliant classifier results file
/// INPUTS:
/// \param out_dir :
/// \param obj_class : The VOC object class identifier string
/// \param dataset : Specifies whether working with the training or test set
/// \param images : An array of ObdImage containing the images for which data will be saved to the result file
/// \param scores : A corresponding array of confidence scores given a query
/// \param competition : Optional
/// \param overwrite_ifexists : Optional
void VOCData::writeClassifierResultsFile(const std::string &out_dir, const std::string &obj_class,
                                         const ObdDatasetType dataset, const std::vector<ObdImage> &images,
                                         const std::vector<float> &scores, const int competition,
                                         const bool overwrite_ifexists) {
    CV_Assert(images.size() == scores.size());

    std::string output_file_base, output_file;
    if (dataset == CV_OBD_TRAIN) {
        output_file_base = out_dir + "/comp" + integerToString(competition) + "_cls_" + m_train_set + "_" + obj_class;
    } else {
        output_file_base = out_dir + "/comp" + integerToString(competition) + "_cls_" + m_test_set + "_" + obj_class;
    }
    output_file = output_file_base + ".txt";

    // check if file exists, and if so create a numbered new file instead
    if (overwrite_ifexists == false) {
        struct stat stFileInfo;
        if (stat(output_file.c_str(), &stFileInfo) == 0) {
            std::string output_file_new;
            int filenum = 0;

            do {
                filenum++;
                output_file_new = output_file_base + "_" + integerToString(filenum);
                output_file = output_file_new + ".txt";
            } while (stat(output_file.c_str(), &stFileInfo) == 0);
        }
    }

    std::ofstream result_file(output_file.c_str());
    if (result_file.is_open()) {
        for (size_t i = 0; i < images.size(); i++) {
            result_file << images[i].id << " " << scores[i] << std::endl;
        }
        result_file.close();
    } else {
        std::string err_msg = "could not open classifier results file '" + output_file +
                              "' for writing. Before running for the first time, a 'results' subdirectory should be created within the VOC dataset base directory. e.g. if the VOC data is stored in /VOC/VOC2010 then the path /VOC/results must be created.";
        CV_Error(CV_StsError, err_msg.c_str());
    }
}

/// Utility function to construct a VOC-standard classification results filename.
/// INPUTS:
/// \param obj_class : The VOC object class identifier string.
/// \param task : Specifies whether to generate a filename for the classification or detection task.
/// \param dataset : Specifies whether working with the training or test set.
/// \param competition : (Optional)
/// \param number : (Optional)
/// \return : The filename of a classification file previously computed using writeClassifierResultFile.
std::string VOCData::getResultsFilename(const std::string &obj_class, const VocTask task, const ObdDatasetType dataset,
                                        const int competition, const int number) {
    if ((competition < 1) && (competition != -1)) {
        CV_Error(CV_StsBadArg, "competition argument should be a positive non-zero number or -1 to accept the default");
    }
    if ((number < 1) && (number != -1)) {
        CV_Error(CV_StsBadArg, "number argument should be a positive non-zero number or -1 to accept the default");
    }

    std::string dset, task_type;

    if (dataset == CV_OBD_TRAIN) {
        dset = m_train_set;
    } else {
        dset = m_test_set;
    }

    int comp = competition;
    if (task == CV_VOC_TASK_CLASSIFICATION) {
        task_type = "cls";
        if (comp == -1)
            comp = 1;
    } else {
        task_type = "det";
        if (comp == -1)
            comp = 3;
    }

    std::stringstream ss;
    if (number < 1) {
        ss << "comp" << comp << "_" << task_type << "_" << dset << "_" << obj_class << ".txt";
    } else {
        ss << "comp" << comp << "_" << task_type << "_" << dset << "_" << obj_class << "_" << number << ".txt";
    }

    std::string filename = ss.str();
    return filename;
}

/// Calculate metrics for classification results
/// INPUTS:
/// \param obj_class : A vector of booleans determining whether the currently tested class is present in each input image
/// \param images :
/// \param scores : A vector containing the similarity score for each input image (higher is more similar)
/// OUTPUTS:
/// \param precision : A vector containing the precision calculated at each datapoint of a p-r curve generated from
///     the result set
/// \param recall : A vector containing the recall calculated at each datapoint of a p-r curve generated from
///     the result set
/// \param ap : The ap metric calculated from the result set
/// \param ranking : Optional
void VOCData::calcClassifierPrecRecall(const std::string &obj_class, const std::vector<ObdImage> &images,
                                       const std::vector<float> &scores, std::vector<float> &precision,
                                       std::vector<float> &recall, float &ap, std::vector<size_t> &ranking) {
    std::vector<char> res_ground_truth;
    getClassifierGroundTruth(obj_class, images, res_ground_truth);

    calcPrecRecall_impl(res_ground_truth, scores, precision, recall, ap, ranking);
}

void VOCData::calcClassifierPrecRecall(const std::string &obj_class, const std::vector<ObdImage> &images,
                                       const std::vector<float> &scores, std::vector<float> &precision,
                                       std::vector<float> &recall, float &ap) {
    std::vector<char> res_ground_truth;
    getClassifierGroundTruth(obj_class, images, res_ground_truth);

    std::vector<size_t> ranking;
    calcPrecRecall_impl(res_ground_truth, scores, precision, recall, ap, ranking);
}

/// Overloaded version which accepts VOC classification result file input instead of array of scores/ground truth
/// INPUTS:
/// \param input_file : The path to the VOC standard results file to use for calculating precision/recall.
///     If a full path is not specified, it is assumed this file is in the VOC standard results directory.
///     A VOC standard filename can be retrieved (as used by writeClassifierResultsFile) by calling getClassifierResultsFilename
/// OUTPUTS:
/// \param precision : A vector containing the precision calculated at each datapoint of a p-r curve generated from
///     the result set
/// \param recall : A vector containing the recall calculated at each datapoint of a p-r curve generated from
///     the result set
/// \param ap : The ap metric calculated from the result set
/// \param outputRankingFile
void VOCData::calcClassifierPrecRecall(const std::string &input_file, std::vector<float> &precision,
                                       std::vector<float> &recall, float &ap, bool outputRankingFile) {
    // read in classification results file
    std::vector<std::string> res_image_codes;
    std::vector<float> res_scores;

    std::string input_file_std = checkFilenamePathsep(input_file);
    readClassifierResultsFile(input_file_std, res_image_codes, res_scores);

    // extract the object class and dataset from the results file filename
    std::string class_name, dataset_name;
    extractDataFromResultsFilename(input_file_std, class_name, dataset_name);

    // generate the ground truth for the images extracted from the results file
    std::vector<char> res_ground_truth;

    getClassifierGroundTruth(class_name, res_image_codes, res_ground_truth);

    if (outputRankingFile) {
        // 1. store sorting order by score (descending) in 'order'
        std::vector<std::pair<size_t, std::vector<float>::const_iterator>> order(res_scores.size());

        size_t n = 0;
        for (std::vector<float>::const_iterator it = res_scores.begin(); it != res_scores.end(); it++, n++) {
            order[n] = make_pair(n, it);
        }
        std::sort(order.begin(), order.end(), orderingSorter());

        // 2. save ranking results to text file
        std::string input_file_std1 = checkFilenamePathsep(input_file);
        size_t fnamestart = input_file_std1.rfind("/");
        std::string scoregt_file_str = input_file_std1.substr(0, fnamestart + 1) + "scoregt_" + class_name + ".txt";
        std::ofstream scoregt_file(scoregt_file_str.c_str());

        if (scoregt_file.is_open()) {
            for (size_t i = 0; i < res_scores.size(); i++) {
                scoregt_file << res_image_codes[order[i].first] << " " << res_scores[order[i].first] << " "
                             << res_ground_truth[order[i].first] << std::endl;
            }
            scoregt_file.close();
        } else {
            std::string err_msg = "could not open scoregt file '" + scoregt_file_str + "' for writing.";
            CV_Error(CV_StsError, err_msg.c_str());
        }
    }

    // finally, calculate precision + recall + ap
    std::vector<size_t> ranking;
    calcPrecRecall_impl(res_ground_truth, res_scores, precision, recall, ap, ranking);
}

void VOCData::calcPrecRecall_impl(const std::vector<char> &ground_truth, const std::vector<float> &scores,
                                  std::vector<float> &precision, std::vector<float> &recall, float &ap,
                                  std::vector<size_t> &ranking, int recall_normalization) {
    CV_Assert(ground_truth.size() == scores.size());

    // add extra element for p-r at 0 recall (in case that first retrieved is positive)
    std::vector<float>(scores.size() + 1).swap(precision);
    std::vector<float>(scores.size() + 1).swap(recall);

    // SORT RESULTS BY THEIR SCORE
    // 1. store sorting order in 'order'
    getSortOrder(scores, ranking);

#ifdef PR_DEBUG
    std::ofstream scoregt_file("./pr.txt");
    if (scoregt_file.is_open()) {
        for (int i = 0; i < scores.size(); ++i) {
            scoregt_file << scores[ranking[i]] << " " << ground_truth[ranking[i]] << std::endl;
        }
        scoregt_file.close();
    }
#endif

    // Calculate precision-recall
    int recall_norm, retrieved_hits = 0;
    if (recall_normalization != -1) {
        recall_norm = recall_normalization;
    } else {
        recall_norm = (int) std::count_if(ground_truth.begin(), ground_truth.end(),
                                          std::bind2nd(std::equal_to<char>(), (char) 1));
    }

    ap = 0;
    recall[0] = 0;
    for (size_t idx = 0; idx < ground_truth.size(); ++idx) {
        if (ground_truth[ranking[idx]] != 0)
            ++retrieved_hits;

        precision[idx + 1] = static_cast<float>(retrieved_hits) / static_cast<float>(idx + 1);
        recall[idx + 1] = static_cast<float>(retrieved_hits) / static_cast<float>(recall_norm);

        if (idx == 0) {
            //add further point at 0 recall with the same precision value as the first computed point
            precision[idx] = precision[idx + 1];
        }
        if (recall[idx + 1] == 1.0) {
            //if recall = 1, then end early as all positive images have been found
            recall.resize(idx + 2);
            precision.resize(idx + 2);
            break;
        }
    }

    /* ap calculation */
    if (m_sampled_ap == false) {
        // FOR VOC2010+ AP IS CALCULATED FROM ALL DATAPOINTS
        /* make precision monotonically decreasing for purposes of calculating ap */
        std::vector<float> precision_monot(precision.size());
        std::vector<float>::iterator prec_m_it = precision_monot.begin();
        for (std::vector<float>::iterator prec_it = precision.begin();
             prec_it != precision.end(); ++prec_it, ++prec_m_it) {
            std::vector<float>::iterator max_elem;
            max_elem = std::max_element(prec_it, precision.end());
            (*prec_m_it) = (*max_elem);
        }
        /* calculate ap */
        for (size_t idx = 0; idx < (recall.size() - 1); ++idx) {
            //no need to take min of prec - is monotonically decreasing
            ap += (recall[idx + 1] - recall[idx]) * precision_monot[idx + 1] +
                  0.5f * (recall[idx + 1] - recall[idx]) * std::abs(precision_monot[idx + 1] - precision_monot[idx]);
        }
    } else {
        // FOR BEFORE VOC2010 AP IS CALCULATED BY SAMPLING PRECISION AT RECALL 0.0,0.1,..,1.0
        for (float recall_pos = 0.f; recall_pos <= 1.f; recall_pos += 0.1f) {
            //find iterator of the precision corresponding to the first recall >= recall_pos
            std::vector<float>::iterator recall_it = recall.begin();
            std::vector<float>::iterator prec_it = precision.begin();

            while ((*recall_it) < recall_pos) {
                ++recall_it;
                ++prec_it;
                if (recall_it == recall.end())
                    break;
            }

            /* if no recall >= recall_pos found, this level of recall is never reached so stop adding to ap */
            if (recall_it == recall.end())
                break;

            /* if the prec_it is valid, compute the max precision at this level of recall or higher */
            std::vector<float>::iterator max_prec = std::max_element(prec_it, precision.end());

            ap += (*max_prec) / 11;
        }
    }
}

/// Calculate rows of a confusion matrix
/// INPUTS:
/// \param obj_class : The VOC object class identifier string for the confusion matrix row to compute
/// \param images : An array of ObdImage containing the images to use for the computation
/// \param scores : A corresponding array of confidence scores for the presence of obj_class in each image
/// \param cond : Defines whether to use a cutoff point based on recall (CV_VOC_CCOND_RECALL)
///     or score (CV_VOC_CCOND_SCORETHRESH) the latter is useful for classifier detections where positive values are
///     positive detections and negative values are negative detections.
/// \param threshold : Threshold value for cond. In case of CV_VOC_CCOND_RECALL, is proportion recall (e.g. 0.5).
///     In the case of CV_VOC_CCOND_SCORETHRESH is the value above which to count result.
/// OUTPUTS:
/// \param output_headers : An output vector of object class headers for the confusion matrix row.
/// \param output_values : An output vector of values for the confusion matrix row corresponding to
///     the classes defined in output_headers
void VOCData::calcClassifierConfMatRow(const std::string &obj_class, const std::vector<ObdImage> &images,
                                       const std::vector<float> &scores, const VocConfCond cond, const float threshold,
                                       std::vector<std::string> &output_headers, std::vector<float> &output_values) {
    if (images.size() != scores.size()) {
        CV_Error(CV_StsAssert, "The number of images and the number of scores doesn't match.");
    }

    // SORT RESULTS BY THEIR SCORE
    // 1. store sorting order in 'ranking'
    std::vector<size_t> ranking;
    getSortOrder(scores, ranking);

    // CALCULATE CONFUSION MATRIX ENTRIES
    // prepare object category headers
    output_headers = m_object_classes;
    std::vector<float>(output_headers.size(), 0.0).swap(output_values);
    // find the index of the target object class in the headers
    int target_idx;
    {
        std::vector<std::string>::iterator target_idx_it = std::find(output_headers.begin(), output_headers.end(),
                                                                     obj_class);
        // if the target class can not be found, raise an exception
        if (target_idx_it == output_headers.end()) {
            std::string err_msg =
                    "could not find the target object class '" + obj_class + "' in list of valid classes.";
            CV_Error(CV_StsError, err_msg.c_str());
        }
        // convert iterator to index
        target_idx = (int) std::distance(output_headers.begin(), target_idx_it);
    }

    // prepare variables related to calculating recall if using the recall threshold
    int retrieved_hits = 0;
    int total_relevant = 0;
    if (cond == CV_VOC_CCOND_RECALL) {
        std::vector<char> ground_truth;
        // in order to calculate the total number of relevant images for normalization of recall,
        // it is necessary to extract the ground truth for the images under consideration
        getClassifierGroundTruth(obj_class, images, ground_truth);
        total_relevant = (int) std::count_if(ground_truth.begin(), ground_truth.end(),
                                             std::bind2nd(std::equal_to<char>(), (char) 1));
    }

    // iterate through images
    std::vector<ObdObject> img_objects;
    std::vector<VocObjectData> img_object_data;
    int total_images = 0;
    for (size_t image_idx = 0; image_idx < images.size(); image_idx++) {
        // if using the score as the break condition, check for it now
        if (cond == CV_VOC_CCOND_SCORETHRESH) {
            if (scores[ranking[image_idx]] <= threshold) {
                break;
            }
        }
        // if continuing for this iteration, increment the image for later normalization
        total_images++;
        // for each image retrieve the objects contained
        getObjects(images[ranking[image_idx]].id, img_objects, img_object_data);
        // check if the tested for object class is present
        if (getClassifierGroundTruthImage(obj_class, images[ranking[image_idx]].id)) {
            // if the target class is present, assign fully to the target class element in the confusion matrix row
            output_values[target_idx] += 1.0;

            if (cond == CV_VOC_CCOND_RECALL) {
                retrieved_hits++;
            }
        } else {
            // first delete all objects marked as difficult
            for (size_t obj_idx = 0; obj_idx < img_objects.size(); obj_idx++) {
                if (img_object_data[obj_idx].difficult == true) {

                    std::vector<ObdObject>::iterator it1 = img_objects.begin();
                    std::advance(it1, obj_idx);
                    img_objects.erase(it1);

                    std::vector<VocObjectData>::iterator it2 = img_object_data.begin();
                    std::advance(it2, obj_idx);
                    img_object_data.erase(it2);

                    obj_idx--;
                }
            }
            // if the target class is not present, add values to the confusion matrix row in equal proportions to
            // all objects present in the image
            for (size_t obj_idx = 0; obj_idx < img_objects.size(); obj_idx++) {
                // find the index of the currently considered object
                std::vector<std::string>::iterator class_idx_it = std::find(output_headers.begin(),
                                                                            output_headers.end(),
                                                                            img_objects[obj_idx].object_class);
                // if the class name extracted from the ground truth file could not be found in the list of available classes, raise an exception
                if (class_idx_it == output_headers.end()) {
                    std::string err_msg = "could not find object class '" + img_objects[obj_idx].object_class +
                                          "' specified in the ground truth file of '" +
                                          images[ranking[image_idx]].id + "'in list of valid classes.";
                    CV_Error(CV_StsError, err_msg.c_str());
                }
                /* convert iterator to index */
                int class_idx = (int) std::distance(output_headers.begin(), class_idx_it);
                //add to confusion matrix row in proportion
                output_values[class_idx] += 1.f / static_cast<float>(img_objects.size());
            }
        }
        //check break conditions if breaking on certain level of recall
        if (cond == CV_VOC_CCOND_RECALL) {
            if (static_cast<float>(retrieved_hits) / static_cast<float>(total_relevant) >= threshold)
                break;
        }
    }
    /* finally, normalize confusion matrix row */
    for (std::vector<float>::iterator it = output_values.begin(); it < output_values.end(); ++it) {
        (*it) /= static_cast<float>(total_images);
    }
}

/// Save Precision-Recall to a p-r curve in GNUPlot format
/// INPUTS:
/// \param output_file : The file to which to save the GNUPlot data file. If only a filename is specified,
///     the data file is saved to the standard VOC results directory
/// \param precision : Vector of precisions as returned from calcClassifier
/// \param recall : Vector of recalls as returned from calcClassifier
/// \param ap : ap as returned from calcClassifier
/// \param title : Title to use for the plot (if not specified, just the ap is printed as the title)
///     This also specifies the filename of the output file if printing to pdf
/// \param plot_type : Specifies whether to instruct GNUPlot to save to a PDF file (CV_VOC_PLOT_PDF) or
///     directly to screen (CV_VOC_PLOT_SCREEN) in the datafile.
void VOCData::savePrecRecallToGnuplot(const std::string &output_file, const std::vector<float> &precision,
                                      const std::vector<float> &recall, const float ap, std::string title,
                                      VocPlotType plot_type) {
    std::string output_file_std = checkFilenamePathsep(output_file);
    std::ofstream plot_file(output_file_std.c_str());

    if (plot_file.is_open())
    {
        plot_file << "set xrange [0:1]" << std::endl;
        plot_file << "set yrange [0:1]" << std::endl;
        plot_file << "set size square" << std::endl;

        std::string title_text = title;
        if (title_text.size() == 0) {
            title_text = "Precision-Recall Curve";
        }

        plot_file << "set title \"" << title_text << " (ap: " << ap << ")\"" << std::endl;
        plot_file << "set xlabel \"Recall\"" << std::endl;
        plot_file << "set ylabel \"Precision\"" << std::endl;
        plot_file << "set style data lines" << std::endl;
        plot_file << "set nokey" << std::endl;
        if (plot_type == CV_VOC_PLOT_PNG)
        {
            plot_file << "set terminal png" << std::endl;
            std::string pdf_filename;
            if (title.size() != 0)
            {
                pdf_filename = title;
            } else {
                pdf_filename = "prcurve";
            }
            plot_file << "set out \"" << title << ".png\"" << std::endl;
        }
        plot_file << "plot \"-\" using 1:2" << std::endl;
        plot_file << "# X Y" << std::endl;
        CV_Assert(precision.size() == recall.size());
        for (size_t i = 0; i < precision.size(); ++i)
        {
            plot_file << "  " << recall[i] << " " << precision[i] << std::endl;
        }
        plot_file << "end" << std::endl;
        if (plot_type == CV_VOC_PLOT_SCREEN)
        {
            plot_file << "pause -1" << std::endl;
        }
        plot_file.close();
    } else {
        std::string err_msg = "could not open plot file '" + output_file_std + "' for writing.";
        CV_Error(CV_StsError,err_msg.c_str());
    }
}

void VOCData::readClassifierGroundTruth(const std::string &obj_class, const ObdDatasetType dataset,
                                        std::vector<ObdImage> &images, std::vector<char> &object_present) {
    images.clear();

    std::string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"), 2, obj_class);
    if (dataset == CV_OBD_TRAIN) {
        gtFilename.replace(gtFilename.find("%s"), 2, m_train_set);
    } else {
        gtFilename.replace(gtFilename.find("%s"), 2, m_test_set);
    }

    std::vector<std::string> image_codes;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);
    convertImageCodesToObdImages(image_codes, images);
}

void VOCData::readClassifierResultsFile(const std::string &input_file, std::vector<ObdImage> &images,
                                        std::vector<float> &scores) {
    images.clear();

    std::string input_file_std = checkFilenamePathsep(input_file);
    std::vector<std::string> image_codes;
    readClassifierResultsFile(input_file_std, image_codes, scores);
    convertImageCodesToObdImages(image_codes, images);
}

const std::vector<std::string> &VOCData::getObjectClasses() {
    return m_object_classes;
}


