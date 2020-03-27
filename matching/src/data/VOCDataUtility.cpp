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

void VOCData::getSortOrder(const std::vector<float> &values, std::vector<size_t> &order, bool descending) {
    // 1. store sorting order in 'order_pair'
    std::vector<std::pair<size_t, std::vector<float>::const_iterator>> order_pair(values.size());

    size_t n = 0;
    for (std::vector<float>::const_iterator it = values.begin(); it != values.end(); it++, n++) {
        order_pair[n] = make_pair(n, it);
    }

    std::sort(order_pair.begin(), order_pair.end(), orderingSorter());
    if (descending == false) {
        std::reverse(order_pair.begin(), order_pair.end());
    }

    std::vector<size_t>(order_pair.size()).swap(order);
    for (size_t i = 0; i < order_pair.size(); i++) {
        order[i] = order_pair[i].first;
    }
}

int VOCData::stringToInteger(const std::string input_str) {
    int result = 0;

    std::stringstream ss(input_str);
    if ((ss >> result).fail()) {
        CV_Error(CV_StsBadArg, "Could not perform string to integer conversion.");
    }
    return result;
}

void VOCData::readFileToString(const std::string filename, std::string &file_contents) {
    std::ifstream ifs(filename.c_str());
    if (!ifs.is_open()) {
        CV_Error(CV_StsError, "Could not open text file.");
    }

    std::stringstream oss;
    oss << ifs.rdbuf();

    file_contents = oss.str();
}

std::string VOCData::integerToString(const int input_int) {
    std::string result;
    std::stringstream ss;

    if ((ss << input_int).fail()) {
        CV_Error(CV_StsBadArg, "Could not perform integer to string conversion.");
    }
    result = ss.str();
    return result;
}

std::string VOCData::checkFilenamePathsep(const std::string filename, bool add_trailing_slash) {
    std::string filename_new = filename;
    size_t pos = filename_new.find("\\\\");

    while (pos != filename_new.npos) {
        filename_new.replace(pos, 2, "/");
        pos = filename_new.find("\\\\", pos);
    }

    pos = filename_new.find("\\");
    while (pos != filename_new.npos) {
        filename_new.replace(pos, 1, "/");
        pos = filename_new.find("\\", pos);
    }

    if (add_trailing_slash) {
        // add training slash if this is missing
        if (filename_new.rfind("/") != filename_new.length() - 1)
            filename_new += "/";
    }
    return filename_new;
}

void VOCData::convertImageCodesToObdImages(const std::vector<std::string> &image_codes, std::vector<ObdImage> &images) {
    images.clear();
    images.reserve(image_codes.size());

    std::string path;
    // transfer to output arrays
    for (size_t i = 0; i < image_codes.size(); i++) {
        // generate image path and indices from extracted string code
        path = getImagePath(image_codes[i]);
        images.push_back(ObdImage(image_codes[i], path));
    }
}

/// Extract text from within a given tag from an XML file.
/// INPUTS:
/// \param src : XML source file
/// \param tag : XML tag delimiting block to extract
/// \param searchpos : position within src at which to start search
/// OUTPUTS:
/// \param tag_contents : text extracted between <tag> and </tag> tags
/// \return : the position of the final character extracted in tag_contents within src (can be used to call
///     extractXMLBlock recursively to extract multiple blocks) returns -1 if the tag could not be found.
int VOCData::extractXMLBlock(const std::string src, const std::string tag, const int searchpos, std::string &tag_contents) {
    size_t startpos, next_startpos, endpos;
    int embed_count = 1;

    // find position of opening tag
    startpos = src.find("<" + tag + ">", searchpos);
    if (startpos == std::string::npos) {
        return -1;
    }

    // initialize endpos - start searching for end tag anywhere after opening tag
    endpos = startpos;
    // find position of next opening tag
    next_startpos = src.find("<" + tag + ">", startpos + 1);
    // match opening tags with closing tags, and only accept final closing tag of same level as original opening tag
    while (embed_count > 0) {
        endpos = src.find("</" + tag + ">", endpos + 1);
        if (endpos == std::string::npos) {
            return -1;
        }
        // the next code is only expected if there are embedded tags with the same name
        if (next_startpos != std::string::npos) {
            while (next_startpos < endpos) {
                // counting embedded start tags
                embed_count++;
                next_startpos = src.find("<" + tag + ">", next_startpos + 1);
                if (next_startpos == std::string::npos) {
                    break;
                }
            }
        }
        // passing end tag so decrement nesting level
        embed_count--;
    }
    // finally, extract the tag region
    startpos += tag.length() + 2;
    if (startpos > src.length())
        return -1;
    if (endpos > src.length())
        return -1;
    tag_contents = src.substr(startpos, endpos - startpos);
    return static_cast<int>(endpos);
}