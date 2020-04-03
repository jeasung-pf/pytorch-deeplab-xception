/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include <matching.grpc.pb.h>
#include <Matcher.hpp>


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using protocol::Feature;
using protocol::Response;
using protocol::Matching;

class MatchingServiceImpl final : public Matching::Service {

    bool create_window(void) {
        cv::namedWindow(m_window_calibrated);
        return true;
    }

    ::grpc::Status calibrate(::grpc::ServerContext *context, const ::protocol::Feature *request, ::protocol::Response *response) {
        std::cout << "Got a new request." << std::endl;

        if (m_model.empty()) {
            std::cout << "Creating a new model..." << std::endl;
            m_feature_params = new cv::Matcher::BRISKFeatureParams();
            m_index_params = new cv::flann::LshIndexParams(10, 10, 2);
            m_search_params = new cv::flann::SearchParams();
            m_model = new cv::Matcher::Matcher(m_feature_params, m_index_params, m_search_params);

//            try {
//                m_model->create_window();
//                std::cout << "Window created..." << std::endl;
//            } catch (std::exception e) {
//                std::cout << e.what() << std::endl;
//                std::cout << "Error creating a namedWindow, check your gui library." << std::endl;
//            }
            m_model->window_created = false;
        }

        if (template_image.empty()) {
            std::cout << "Reading a template image" << std::endl;
            template_image = cv::imread("../data/identification.jpg");
        }

        std::string image_encoded = request->image_encoded();
        std::string image_segmentation_class_encoded = request->image_segmentation_class_encoded();
        cv::Mat image, image_segmentation_class, calibrated;

        try {
            std::vector<uchar> image_buffer(image_encoded.begin(), image_encoded.end()),
                    image_segmentation_class_buffer(image_segmentation_class_encoded.begin(),
                                                    image_segmentation_class_encoded.end());
            image = cv::imdecode(image_buffer, cv::IMREAD_COLOR);
            image_segmentation_class = cv::imdecode(image_segmentation_class_buffer, cv::IMREAD_GRAYSCALE);
        } catch (std::exception e) {
            std::cout << e.what() << std::endl;
            std::cout << "Error while decoding a received image" << std::endl;
        }

        m_model->train(image, template_image, image_segmentation_class, calibrated, 128.0);

        // cv::imshow(m_window_calibrated, calibrated);
        cv::imwrite("./response.jpg", calibrated);

        response->set_image_filename("");
        response->set_image_recognition_dates("");
        response->set_image_recognition_numbers("");
        response->set_image_recognition_verification("");

        return Status::OK;
    }

protected:
    cv::Ptr<cv::Matcher::Matcher> m_model;
    cv::Ptr<cv::Matcher::FeatureParams> m_feature_params;
    cv::Ptr<cv::flann::IndexParams> m_index_params;
    cv::Ptr<cv::flann::SearchParams> m_search_params;
    cv::Mat template_image;

    std::string m_window_calibrated = "Calibrated image";

};

void RunServer(void) {
    std::string server_address("0.0.0.0:50052");
    MatchingServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char **argv) {
    RunServer();

    return 0;
}
